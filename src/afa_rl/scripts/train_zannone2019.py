import argparse

import torch
import yaml
from jaxtyping import Float
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torchrl.collectors import SyncDataCollector
from torchrl.envs import ExplorationType, set_exploration_type
from tqdm import tqdm

import wandb
from afa_rl.afa_env import AFAEnv, Shim2018Env
from afa_rl.afa_methods import Shim2018AFAMethod, Zannone2019AFAMethod
from afa_rl.agents import Shim2018Agent, Zannone2019Agent
from afa_rl.datasets import get_afa_dataset_fn
from afa_rl.models import (
    MLPClassifier,
    PartialVAE,
    ReadProcessEncoder,
    ShimEmbedder,
    ShimEmbedderClassifier,
)
from afa_rl.scripts.pretrain_zannone2019 import get_zannone2019_model_from_config
from afa_rl.utils import FloatWrapFn, dict_to_namespace, get_sequential_module_norm
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
from common.datasets import CubeDataset


def get_zannone2019_reward_fn(
    partial_vae: PartialVAE,
    classifier: nn.Module,
    acquisition_costs: Float[Tensor, "batch n_features"],
) -> AFARewardFn:
    """
    Returns the reward function as defined in "ODIN: Optimal Discovery of High-value INformation Using Model-based Deep Reinforcement Learning"
    """

    def f(
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        new_masked_features: MaskedFeatures,
        new_feature_mask: FeatureMask,
        afa_selection: AFASelection,
        features: Features,
        label: Label,
    ) -> AFAReward:
        reward = torch.zeros_like(afa_selection, dtype=torch.float32)
        is_done = afa_selection.squeeze(-1) == 0

        # If episode is finished, use negative classification loss as reward

        # The classifier expects to act on the latent space, so find the latent representation of the masked features but only pick the mean
        encoding, mu, logvar, z = partial_vae.encode(
            new_masked_features, new_feature_mask
        )

        # Get logits using classifier
        logits = classifier(mu)

        # Loss is cross-entropy loss
        loss = F.cross_entropy(logits, label)
        reward[is_done] = -loss

        # If episode is not finished, use negative acquisition cost as reward
        acquisition_cost = acquisition_costs[afa_selection.squeeze(-1) - 1].sum()
        reward[~is_done] = -acquisition_cost

        return reward

    return f


def main(args):
    torch.set_float32_matmul_precision("medium")

    # Load configs from yaml files
    with open(args.pretrain_config, "r") as file:
        pretrain_config_dict: dict = yaml.safe_load(file)
    with open(args.train_config, "r") as file:
        train_config_dict: dict = yaml.safe_load(file)

    pretrain_config = dict_to_namespace(pretrain_config_dict)
    train_config = dict_to_namespace(train_config_dict)

    device = torch.device(train_config.device)

    # Load pretrained model
    pretrained_model = get_zannone2019_model_from_config(pretrain_config)

    # Generate data from generative model
    z = torch.randn(
        train_config.generative_batch_size, pretrain_config.partial_vae.latent_size
    )
    xhat = pretrained_model.partial_vae.decoder(z)
    # Predicted labels
    yhat = pretrained_model.classifier(z)

    # Prepare cube dataset for the format that AFAMDP expects
    dataset = CubeDataset(
        n_features=pretrain_config.n_features,
        data_points=train_config.dataset.size,
        sigma=train_config.dataset.sigma,
        seed=train_config.dataset.seed,
    )
    dataset.generate_data()

    # Concatenate the generated data with the original dataset
    combined_features = torch.cat((dataset.features, xhat), dim=0)
    combined_labels = torch.cat((dataset.labels, yhat), dim=0)

    reward_fn = get_zannone2019_reward_fn(
        partial_vae=pretrained_model.partial_vae,
        classifier=pretrained_model.classifier,
        acquisition_costs=torch.tensor(
            data=train_config.acquisition_costs,
            dtype=torch.float32,
            device=device,
        ),
    )

    # Use original data and generated data for training
    train_env = AFAEnv(
        dataset_fn=get_afa_dataset_fn(combined_features, combined_labels),
        reward_fn=reward_fn,
        device=device,
        batch_size=train_config.batch_size,
    )
    # check_env_specs(train_env)

    # Evaluate only on original data
    eval_env = AFAEnv(
        dataset_fn=get_afa_dataset_fn(dataset.features, dataset.labels),
        reward_fn=reward_fn,
        device=device,
        batch_size=train_config.batch_size,
    )

    agent = Zannone2019Agent(
        partial_vae=pretrained_model.partial_vae,
        lr=train_config.agent.lr,
        update_tau=train_config.agent.update_tau,
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
                    "train/batch_idx": batch_idx,
                }
            )

            if batch_idx != 0 and batch_idx % config.eval_every_n_batches == 0:
                with (
                    torch.no_grad(),
                    set_exploration_type(ExplorationType.DETERMINISTIC),
                ):
                    eval_metrics = {
                        "acquired_features": torch.zeros(
                            config.eval_episodes,
                            dtype=torch.int64,
                            device=config.device,
                        ),
                        "is_correct_class": torch.zeros(
                            config.eval_episodes, dtype=torch.bool, device=config.device
                        ),
                        "predicted_classes": torch.zeros(
                            config.eval_episodes,
                            dtype=torch.int64,
                            device=config.device,
                        ),
                        "reward": torch.zeros(
                            config.eval_episodes,
                            dtype=torch.float32,
                            device=config.device,
                        ),
                        "traj_len": torch.zeros(
                            config.eval_episodes,
                            dtype=torch.int64,
                            device=config.device,
                        ),
                    }
                    for eval_episode in tqdm(
                        range(train_config.eval_episodes), desc="Evaluating agent..."
                    ):
                        # Evaluate agent
                        td_eval = eval_env.rollout(
                            max_steps=train_config.eval_max_steps, policy=agent.policy
                        )
                        # Squeeze batch dimension
                        td_eval = td_eval.squeeze(0)
                        # Calculate acquired features
                        eval_metrics["acquired_features"][eval_episode] = td_eval[
                            "feature_mask"
                        ][-1].sum()
                        # Check whether classification was correct
                        encoding, mu, logvar, z = pretrained_model.partial_vae.encode(
                            td_eval["masked_features"][-1], td_eval["feature_mask"][-1]
                        )
                        predicted_class = pretrained_model.classifier(mu).argmax(dim=-1)
                        eval_metrics["predicted_classes"][eval_episode] = (
                            predicted_class
                        )
                        eval_metrics["is_correct_class"][eval_episode] = (
                            predicted_class == td_eval["label"][-1].argmax(dim=-1)
                        )
                        eval_metrics["reward"][eval_episode] = td_eval[
                            "next", "reward"
                        ].mean()
                        eval_metrics["traj_len"][eval_episode] = len(td_eval)
                    wandb.log(
                        {
                            "eval/acquired_features": eval_metrics["acquired_features"]
                            .float()
                            .mean()
                            .item(),
                            "eval/accuracy": eval_metrics["is_correct_class"]
                            .float()
                            .mean()
                            .item(),
                            "eval/predicted_class": wandb.Histogram(
                                eval_metrics["predicted_classes"].cpu().numpy(),
                            ),
                            "eval/reward": eval_metrics["reward"].mean().item(),
                            "eval/traj_len": eval_metrics["traj_len"]
                            .float()
                            .mean()
                            .item(),
                            "eval/batch_idx": batch_idx,
                        }
                    )
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
