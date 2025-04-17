import argparse

import torch
import yaml
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F
from torchrl.collectors import SyncDataCollector
from torchrl.envs import ExplorationType, set_exploration_type
from tqdm import tqdm

import wandb
from afa_rl.afa_env import AFAEnv, get_zannone2019_reward_fn
from afa_rl.afa_methods import Zannone2019AFAMethod
from afa_rl.agents import Zannone2019Agent
from afa_rl.datasets import get_afa_dataset_fn
from afa_rl.models import (
    PartialVAE,
    Zannone2019PretrainingModel,
)
from afa_rl.scripts.pretrain_zannone2019 import get_zannone2019_model_from_config
from afa_rl.utils import dict_to_namespace, get_sequential_module_norm
from common.custom_types import (
    AFADataset,
    AFAReward,
    AFARewardFn,
    AFASelection,
    FeatureMask,
    Features,
    Label,
    MaskedFeatures,
)
from common.registry import AFA_DATASET_REGISTRY


def get_pretrained_model_accuracy(
    pretrained_model: Zannone2019PretrainingModel,
    dataset: AFADataset,
    mask_probability: float,
    n_samples=100,
) -> float:
    """
    Return accuracy of the pretrained model on a masked dataset, using n_samples samples.
    """

    # Sample n_samples from the train and val datasets
    indices = torch.randint(0, len(dataset), (n_samples,))
    features, labels = dataset[indices]
    # Create a feature mask with mask_probability
    feature_mask = torch.rand(features.shape) > mask_probability
    # Mask the features
    masked_features = features * feature_mask
    # Get the latent representation of the masked features
    encoding, mu, logvar, z = pretrained_model.partial_vae.encode(
        masked_features, feature_mask
    )
    # Get logits using classifier
    logits = pretrained_model.classifier(mu)
    # Calculate accuracy
    _, predicted_labels = logits.max(dim=-1)
    _, true_labels = labels.max(dim=-1)
    accuracy = (predicted_labels == true_labels).float().mean()
    return accuracy


def train_log(run, td, agent, agent_loss, batch_idx):
    run.log(
        {
            "train/agent_loss": agent_loss,
            # "train/action": td["action"].item(),
            "train/reward": td["next", "reward"].mean().item(),
            # "train/episode_idx": episode_idx,
            "train/acquired_features": td["feature_mask"][
                td["next", "done"].squeeze(-1)
            ]
            .sum(dim=-1)
            .float()
            .mean()
            .item(),
            # "train/accuracy": td["label"][td["next", "done"].squeeze(-1)]
            # .argmax(dim=-1)
            # .eq(td["next", "predicted_class"][td["next", "done"].squeeze(-1)])
            # .float()
            # .mean()
            # .item(),
            # "train/replay buffer size": len(agent.replay_buffer),
            "train/agent_value_net_norm": get_sequential_module_norm(
                agent.value_module.net
            ),
            "train/agent_policy_net_norm": get_sequential_module_norm(
                agent.policy_module.net
            ),
            "train/batch_idx": batch_idx,
        }
    )


def get_eval_metrics(train_config, eval_env, agent, pretrained_model):
    eval_metrics = {
        "acquired_features": torch.zeros(
            train_config.eval_episodes,
            dtype=torch.int64,
            device=train_config.device,
        ),
        "is_correct_class": torch.zeros(
            train_config.eval_episodes, dtype=torch.bool, device=train_config.device
        ),
        "predicted_classes": torch.zeros(
            train_config.eval_episodes,
            dtype=torch.int64,
            device=train_config.device,
        ),
        "reward": torch.zeros(
            train_config.eval_episodes,
            dtype=torch.float32,
            device=train_config.device,
        ),
        "traj_len": torch.zeros(
            train_config.eval_episodes,
            dtype=torch.int64,
            device=train_config.device,
        ),
    }
    for eval_episode in tqdm(
        range(train_config.eval_episodes), desc="Evaluating agent..."
    ):
        # Evaluate agent
        td_eval = eval_env.rollout(
            max_steps=train_config.eval_max_steps, policy=agent.policy
        )
        # TODO: decide whether rollouts can be longer than n_features
        # Squeeze batch dimension
        td_eval = td_eval.squeeze(0)
        # Calculate acquired features
        eval_metrics["acquired_features"][eval_episode] = td_eval["feature_mask"][
            -1
        ].sum()
        # Check whether classification was correct
        _, mu, _, _ = pretrained_model.partial_vae.encode(
            td_eval["masked_features"][-1:], td_eval["feature_mask"][-1:]
        )
        predicted_class = pretrained_model.classifier(mu).argmax(dim=-1).squeeze(0)
        eval_metrics["predicted_classes"][eval_episode] = predicted_class
        eval_metrics["is_correct_class"][eval_episode] = predicted_class == td_eval[
            "label"
        ][-1].argmax(dim=-1)
        eval_metrics["reward"][eval_episode] = td_eval["next", "reward"].mean()
        eval_metrics["traj_len"][eval_episode] = len(td_eval)
    return eval_metrics


def eval_log(run, eval_metrics, batch_idx):
    run.log(
        {
            "eval/acquired_features": eval_metrics["acquired_features"]
            .float()
            .mean()
            .item(),
            "eval/accuracy": eval_metrics["is_correct_class"].float().mean().item(),
            "eval/predicted_class": wandb.Histogram(
                eval_metrics["predicted_classes"].cpu().numpy(),
            ),
            "eval/reward": eval_metrics["reward"].mean().item(),
            "eval/traj_len": eval_metrics["traj_len"].float().mean().item(),
            "eval/batch_idx": batch_idx,
        }
    )


def main(args):
    torch.set_float32_matmul_precision("medium")

    # Load configs from yaml files
    with open(args.pretrain_config, "r") as file:
        pretrain_config_dict: dict = yaml.safe_load(file)
    with open(args.train_config, "r") as file:
        train_config_dict: dict = yaml.safe_load(file)

    pretrain_config = dict_to_namespace(pretrain_config_dict)
    train_config = dict_to_namespace(train_config_dict)
    assert (
        train_config.n_generative_samples % train_config.generative_batch_size == 0
    ), (
        "Number of generative samples must be divisible by the generative batch size"
        f" (n_generative_samples: {train_config.n_generative_samples}, generative_batch_size: {train_config.generative_batch_size})"
    )

    device = torch.device(train_config.device)

    # Init pretrained model
    pretrained_model = get_zannone2019_model_from_config(pretrain_config)
    # Load checkpoint
    pretrained_model_checkpoint = torch.load(
        pretrain_config.checkpoint_path, weights_only=True
    )
    pretrained_model.load_state_dict(pretrained_model_checkpoint["state_dict"])
    pretrained_model = pretrained_model.to(device)
    # Freeze it
    pretrained_model.requires_grad_(False)
    pretrained_model.eval()

    # Load datasets
    train_dataset: AFADataset = AFA_DATASET_REGISTRY[train_config.dataset.name].load(
        train_config.dataset.train_path
    )
    val_dataset: AFADataset = AFA_DATASET_REGISTRY[train_config.dataset.name].load(
        train_config.dataset.val_path
    )

    # Verify that the model has decent performance on the datasets
    train_acc_50 = get_pretrained_model_accuracy(
        pretrained_model, train_dataset, 0.5, n_samples=100
    )
    train_acc_100 = get_pretrained_model_accuracy(
        pretrained_model, train_dataset, 0.0, n_samples=100
    )
    val_acc_50 = get_pretrained_model_accuracy(
        pretrained_model, val_dataset, 0.5, n_samples=100
    )
    val_acc_100 = get_pretrained_model_accuracy(
        pretrained_model, val_dataset, 0.0, n_samples=100
    )
    print(f"Train accuracy (50% mask): {train_acc_50:.4f}")
    print(f"Train accuracy (100% mask): {train_acc_100:.4f}")
    print(f"Val accuracy (50% mask): {val_acc_50:.4f}")
    print(f"Val accuracy (100% mask): {val_acc_100:.4f}")

    # Generate data from generative model, using specified batch size
    xhats = []
    yhats = []
    for _ in range(
        train_config.n_generative_samples // train_config.generative_batch_size
    ):
        z = torch.randn(
            train_config.generative_batch_size, pretrain_config.partial_vae.latent_size
        )
        xhat = pretrained_model.partial_vae.decoder(z)
        xhats.append(xhat)
        # Predicted labels, keeping all probabilities!
        yhat = F.softmax(pretrained_model.classifier(z), -1)
        yhats.append(yhat)

    # Concatenate the generated data with the original dataset
    combined_features = torch.cat((train_dataset.features, xhat), dim=0)
    combined_labels = torch.cat((train_dataset.labels, yhat), dim=0)

    reward_fn = get_zannone2019_reward_fn(
        partial_vae=pretrained_model.partial_vae,
        classifier=pretrained_model.classifier,
        acquisition_costs=train_config.acquisition_cost
        * torch.ones(train_dataset.features.shape[1], dtype=torch.float32),
    )

    # Use original data and generated data for training
    train_env = AFAEnv(
        dataset_fn=get_afa_dataset_fn(combined_features, combined_labels),
        reward_fn=reward_fn,
        device=device,
        batch_size=torch.Size((train_config.n_agents,)),
        feature_size=train_dataset.features.shape[1],
        n_classes=train_dataset.labels.shape[1],
    )
    td = train_env.reset()
    td = train_env.rand_step(td)
    # check_env_specs(train_env)

    # Evaluate on validation data
    eval_env = AFAEnv(
        dataset_fn=get_afa_dataset_fn(val_dataset.features, val_dataset.labels),
        reward_fn=reward_fn,
        device=device,
        batch_size=torch.Size((1,)),
        feature_size=val_dataset.features.shape[1],
        n_classes=val_dataset.labels.shape[1],
    )

    agent = Zannone2019Agent(
        pointnet=pretrained_model.partial_vae.pointnet,
        encoder=pretrained_model.partial_vae.encoder,
        lr=train_config.agent.lr,
        device=device,
        latent_size=pretrain_config.partial_vae.latent_size,
        action_spec=train_env.action_spec,
    )

    collector = SyncDataCollector(
        train_env,
        agent.policy,
        frames_per_batch=train_config.batch_size,
        total_frames=train_config.n_batches * train_config.batch_size,
    )

    # Use WandB for logging
    run = wandb.init(
        entity=train_config.wandb.entity,
        project=train_config.wandb.project,
    )

    # Training loop
    try:
        for batch_idx, td in tqdm(
            enumerate(collector), total=train_config.n_batches, desc="Training agent..."
        ):
            # Collapse agent and batch dimensions
            td = td.flatten(start_dim=0, end_dim=1)
            agent_loss = agent.process_batch(td)

            # Logging
            train_log(run, td, agent, agent_loss, batch_idx)

            # Evaluation sometimes
            if batch_idx != 0 and batch_idx % train_config.eval_every_n_batches == 0:
                with (
                    torch.no_grad(),
                    set_exploration_type(ExplorationType.DETERMINISTIC),
                ):
                    eval_metrics = get_eval_metrics(
                        train_config, eval_env, agent, pretrained_model
                    )
                eval_log(run, eval_metrics, batch_idx)
    except KeyboardInterrupt:
        pass

    # Convert the embedder+agent to an AFAMethod and save it
    afa_method = Zannone2019AFAMethod(agent.actor_network)
    afa_method.save(f"models/afa_rl/{train_config.afa_method_save_name}")


if __name__ == "__main__":
    # Use argparse to choose config file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrain-config",
        type=str,
        required=True,
        help="Path to YAML config file for pretraining",
    )
    parser.add_argument(
        "--train-config",
        type=str,
        required=True,
        help="Path to YAML config file for training",
    )
    args = parser.parse_args()

    main(args)
