import argparse
import copy
from functools import partial
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F
from torchrl.collectors import SyncDataCollector
from torchrl.envs import ExplorationType, check_env_specs, set_exploration_type
from tqdm import tqdm

import wandb
from afa_rl.afa_env import AFAEnv, get_common_reward_fn
from afa_rl.zannone2019.afa_methods import Zannone2019AFAMethod
from afa_rl.zannone2019.agents import Zannone2019Agent
from afa_rl.custom_types import MaskedClassifier, Logits
from afa_rl.datasets import get_afa_dataset_fn
from afa_rl.zannone2019.models import (
    Zannone2019PretrainingModel,
)
from afa_rl.zannone2019.scripts.pretrain_zannone2019 import (
    get_zannone2019_model_from_config,
)
from afa_rl.utils import get_sequential_module_norm
from common.utils import dict_to_namespace
from common.custom_types import (
    AFADataset,
    FeatureMask,
    Features,
    Label,
    MaskedFeatures,
)
from common.utils import get_class_probabilities, set_seed


def get_pretrained_model_accuracy(
    pretrained_model: Zannone2019PretrainingModel,
    features: Features,
    labels: Label,
    mask_probability: float,
    n_samples=100,
) -> float:
    """
    Return accuracy of the pretrained model on a masked dataset, using n_samples samples.

    Assumes there is enough space on pretrained_model.device to hold the entire dataset.
    """

    # Sample n_samples from the train and val datasets
    indices = torch.randperm(features.shape[0])[:n_samples]
    sampled_features = features[indices]
    sampled_labels = labels[indices]
    # Create a feature mask with mask_probability
    feature_mask = torch.rand(sampled_features.shape) > mask_probability
    # Mask the features
    masked_features = sampled_features * feature_mask
    # Get the latent representation of the masked features
    with torch.no_grad():
        encoding, mu, logvar, z = pretrained_model.partial_vae.encode(
            masked_features.to(pretrained_model.device),
            feature_mask.to(pretrained_model.device),
        )
        # Get logits using classifier
        logits = pretrained_model.classifier(mu.to(pretrained_model.device))
    logits = logits.cpu()
    # Calculate accuracy
    _, predicted_labels = logits.max(dim=-1)
    _, true_labels = sampled_labels.max(dim=-1)
    accuracy = (predicted_labels == true_labels).float().mean()
    return accuracy


def train_log(run, td, agent, agent_loss, batch_idx):
    run.log(
        {
            "train/agent_loss": agent_loss,
            # "train/action": td["action"].item(),
            "train/reward": td["next", "reward"].mean().item(),
            # "train/episode_idx": episode_idx,
            "train/agent_value_head_norm": get_sequential_module_norm(
                agent.value_head.net
            ),
            "train/agent_policy_head_norm": get_sequential_module_norm(
                agent.policy_head.net
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
        "reward sum": torch.zeros(
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
        td_eval = eval_env.rollout(max_steps=9001, policy=agent.policy)
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
        eval_metrics["reward sum"][eval_episode] = td_eval["next", "reward"].sum()
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
            "eval/reward sum": eval_metrics["reward sum"].mean().item(),
            "eval/traj_len": eval_metrics["traj_len"].float().mean().item(),
            "eval/batch_idx": batch_idx,
        }
    )


def check_pretrained_model_accuracy(
    pretrained_model, train_dataset, val_dataset, gen_features, gen_labels
) -> None:
    train_acc_50 = get_pretrained_model_accuracy(
        pretrained_model,
        train_dataset.features,
        train_dataset.labels,
        0.5,
        n_samples=len(train_dataset),
    )
    train_acc_100 = get_pretrained_model_accuracy(
        pretrained_model,
        train_dataset.features,
        train_dataset.labels,
        0.0,
        n_samples=len(train_dataset),
    )
    val_acc_50 = get_pretrained_model_accuracy(
        pretrained_model,
        val_dataset.features,
        val_dataset.labels,
        0.5,
        n_samples=len(val_dataset),
    )
    val_acc_100 = get_pretrained_model_accuracy(
        pretrained_model,
        val_dataset.features,
        val_dataset.labels,
        0.0,
        n_samples=len(val_dataset),
    )
    gen_acc_50 = get_pretrained_model_accuracy(
        pretrained_model, gen_features, gen_labels, 0.5, n_samples=len(gen_features)
    )
    gen_acc_100 = get_pretrained_model_accuracy(
        pretrained_model, gen_features, gen_labels, 0.0, n_samples=len(gen_features)
    )
    print(f"Train accuracy (50% mask): {train_acc_50:.4f}")
    print(f"Train accuracy (100% mask): {train_acc_100:.4f}")
    print(f"Val accuracy (50% mask): {val_acc_50:.4f}")
    print(f"Val accuracy (100% mask): {val_acc_100:.4f}")
    print(f"Gen accuracy (50% mask): {gen_acc_50:.4f}")
    print(f"Gen accuracy (100% mask): {gen_acc_100:.4f}")


class Zannone2019AFAClassifier(MaskedClassifier):
    def __init__(self, model: Zannone2019PretrainingModel):
        self.model = model

    def __call__(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Logits:
        with torch.no_grad():
            encoding, mu, logvar, z = self.model.partial_vae.encode(
                masked_features, feature_mask
            )
            logits = self.model.classifier(mu)
        return logits


def assert_deterministic_pretrained_model(pretrained_model, features):
    z1 = pretrained_model.partial_vae.encoder(
        features.to(pretrained_model.device),
        torch.ones_like(features).to(pretrained_model.device),
    )
    mu1 = z1[:, : z1.shape[1] // 2]
    logits1 = pretrained_model.classifier(mu1)
    z2 = pretrained_model.partial_vae.encoder(
        features.to(pretrained_model.device),
        torch.ones_like(features).to(pretrained_model.device),
    )
    mu2 = z2[:, : z2.shape[1] // 2]
    logits2 = pretrained_model.classifier(mu2)
    assert torch.allclose(z1, z2)
    assert torch.allclose(logits1, logits2)


def main(
    pretrain_config_path: Path,
    train_config_path: Path,
    dataset_type: str,
    train_dataset_path: Path,
    val_dataset_path: Path,
    pretrained_model_path: Path,
    hard_budget: int,
    seed: int,
    afa_method_path: Path,
) -> None:
    set_seed(seed)
    torch.set_float32_matmul_precision("medium")

    # Load configs from yaml files
    with open(pretrain_config_path, "r") as file:
        pretrain_config_dict: dict = yaml.safe_load(file)
    with open(train_config_path, "r") as file:
        train_config_dict: dict = yaml.safe_load(file)

    pretrain_config = dict_to_namespace(pretrain_config_dict)
    train_config = dict_to_namespace(train_config_dict)
    device = torch.device(train_config.device)

    # Load datasets
    # Import is delayed until now to avoid circular imports
    from common.registry import AFA_DATASET_REGISTRY

    train_dataset: AFADataset = AFA_DATASET_REGISTRY[dataset_type].load(
        train_dataset_path
    )
    val_dataset: AFADataset = AFA_DATASET_REGISTRY[dataset_type].load(val_dataset_path)
    print(f"Old dataset size: {train_dataset.features.shape[0]}")

    # Get the number of features and classes from the dataset
    n_features = train_dataset.features.shape[-1]
    n_classes = train_dataset.labels.shape[-1]

    class_probabilities = get_class_probabilities(train_dataset.labels)

    # Init pretrained model
    pretrained_model = get_zannone2019_model_from_config(
        pretrain_config, n_features, n_classes, class_probabilities
    )
    # Load checkpoint
    pretrained_model_checkpoint = torch.load(pretrained_model_path, weights_only=True)
    pretrained_model.load_state_dict(pretrained_model_checkpoint["state_dict"])
    pretrained_model = pretrained_model.to(device)
    # Freeze it
    pretrained_model.requires_grad_(False)
    pretrained_model.eval()

    # We want a deterministic model
    assert_deterministic_pretrained_model(pretrained_model, val_dataset.features)

    # Generate data from generative model, using specified batch size
    # Always generate as much data as the original dataset, making it twice as large
    xhats = []
    yhats = []
    for _ in range(len(train_dataset) // train_config.generative_batch_size):
        z = torch.randn(
            train_config.generative_batch_size, pretrain_config.partial_vae.latent_size
        )
        with torch.no_grad():
            xhat = pretrained_model.partial_vae.decoder(
                z.to(pretrained_model.device)
            ).cpu()
        xhats.append(xhat)
        # Predicted labels, keeping all probabilities!
        with torch.no_grad():
            yhat = F.softmax(
                pretrained_model.classifier(z.to(pretrained_model.device)), -1
            ).cpu()
        yhats.append(yhat)
    # There might be a remainder, so we need to pad the last batch
    if len(train_dataset) % train_config.generative_batch_size != 0:
        z = torch.randn(
            len(train_dataset) % train_config.generative_batch_size,
            pretrain_config.partial_vae.latent_size,
        )
        with torch.no_grad():
            xhat = pretrained_model.partial_vae.decoder(
                z.to(pretrained_model.device)
            ).cpu()
        xhats.append(xhat)
        # Predicted labels, keeping all probabilities!
        with torch.no_grad():
            yhat = F.softmax(
                pretrained_model.classifier(z.to(pretrained_model.device)), -1
            ).cpu()
        yhats.append(yhat)

    xhats = torch.cat(xhats, dim=0)
    yhats = torch.cat(yhats, dim=0)

    # Verify that the model has decent performance on the datasets
    check_pretrained_model_accuracy(
        pretrained_model, train_dataset, val_dataset, xhats, yhats
    )

    # Concatenate the generated data with the original dataset
    combined_train_features = torch.cat((train_dataset.features, xhats), dim=0)
    combined_train_labels = torch.cat((train_dataset.labels, yhats), dim=0)
    # Shuffle once in the beginning, but is also done after each epoch
    indices = torch.randperm(combined_train_features.shape[0])
    combined_train_features = combined_train_features[indices]
    combined_train_labels = combined_train_labels[indices]
    print(f"New dataset size: {combined_train_features.shape[0]}")

    # Zannone2019 reward function
    class_weights = 1.0 / class_probabilities
    class_weights = class_weights / class_weights.sum()
    class_weights_device = class_weights.to(device)
    reward_fn = get_common_reward_fn(
        classifier=Zannone2019AFAClassifier(pretrained_model),
        loss_fn=partial(F.cross_entropy, weight=class_weights_device),
    )

    # MDP expects special dataset functions
    train_dataset_fn = get_afa_dataset_fn(
        combined_train_features, combined_train_labels
    )
    val_dataset_fn = get_afa_dataset_fn(val_dataset.features, val_dataset.labels)

    # Use original data and generated data for training
    train_env = AFAEnv(
        dataset_fn=train_dataset_fn,
        reward_fn=reward_fn,
        device=device,
        batch_size=torch.Size((train_config.n_agents,)),
        feature_size=n_features,
        n_classes=n_classes,
        hard_budget=hard_budget,
    )
    check_env_specs(train_env)

    # Evaluate on validation data
    eval_env = AFAEnv(
        dataset_fn=val_dataset_fn,
        reward_fn=reward_fn,
        device=device,
        batch_size=torch.Size((1,)),
        feature_size=n_features,
        n_classes=n_classes,
        hard_budget=hard_budget,
    )

    agent = Zannone2019Agent(
        pointnet=pretrained_model.partial_vae.pointnet,
        encoder=pretrained_model.partial_vae.encoder,
        lr=train_config.agent.lr,
        device=device,
        latent_size=pretrain_config.partial_vae.latent_size,
        action_spec=train_env.action_spec,
        lmbda=train_config.agent.lmbda,
        clip_epsilon=train_config.agent.clip_epsilon,
        entropy_bonus=train_config.agent.entropy_bonus,
        entropy_coef=train_config.agent.entropy_coef,
        max_grad_norm=train_config.agent.max_grad_norm,
    )

    collector = SyncDataCollector(
        train_env,
        agent.policy,
        frames_per_batch=train_config.batch_size,
        total_frames=train_config.n_batches * train_config.batch_size,
        device=device,
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
    finally:
        run.finish()

        # Convert the embedder+agent to an AFAMethod and save it
        afa_method = Zannone2019AFAMethod(
            device, agent.probabilistic_policy_tdmodule, pretrained_model
        )
        afa_method.save(afa_method_path / "model.pt")
        print(f"Zannone2019AFAMethod saved to {afa_method_path / 'model.pt'}")

        # Save params.yml file
        with open(afa_method_path / "params.yml", "w") as file:
            yaml.dump(
                {
                    "hard_budget": hard_budget,
                    "seed": seed,
                    "dataset_type": dataset_type,
                    "train_dataset_path": str(train_dataset_path),
                    "val_dataset_path": str(val_dataset_path),
                    "pretrained_model_path": str(pretrained_model_path),
                },
                file,
            )

        # Load it just to check that it works
        afa_method = Zannone2019AFAMethod.load(afa_method_path, device)
        print(f"Zannone2019AFAMethod loaded from {afa_method_path}")
        check_pretrained_model_accuracy(
            afa_method.pretrained_model, train_dataset, val_dataset, xhats, yhats
        )


if __name__ == "__main__":
    # Use argparse to choose config file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrain_config_path",
        type=str,
        required=True,
        help="Path to YAML config file for pretraining",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        required=True,
        help="Path to YAML config file for training",
    )
    parser.add_argument(
        "--dataset_type", type=str, required=True, choices=AFA_DATASET_REGISTRY.keys()
    )
    parser.add_argument("--train_dataset_path", type=Path, required=True)
    parser.add_argument("--val_dataset_path", type=Path, required=True)
    parser.add_argument("--pretrained_model_path", type=Path, required=True)
    parser.add_argument(
        "--afa_method_path",
        type=Path,
        required=True,
        help="Path to folder to save the trained AFA method",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--hard_budget", type=int, required=True)
    args = parser.parse_args()

    main(
        args.pretrain_config_path,
        args.train_config,
        args.dataset_type,
        args.train_dataset_path,
        args.val_dataset_path,
        args.pretrained_model_path,
        args.afa_method_path,
        args.seed,
        args.hard_budget,
    )
