import argparse
from functools import partial
from pathlib import Path
from typing import Any

import torch
import yaml
from jaxtyping import Float
from tensordict import TensorDictBase
from torch import Tensor, optim
from torch.nn import functional as F
from torchrl.collectors import SyncDataCollector
from torchrl.envs import ExplorationType, check_env_specs, set_exploration_type
from torchrl_agents import Agent
from tqdm import tqdm

import wandb
from afa_rl.afa_env import AFAEnv, get_common_reward_fn
from afa_rl.afa_methods import RLAFAMethod
from afa_rl.shim2018.agents import Shim2018Agent
from afa_rl.custom_types import MaskedClassifier
from afa_rl.datasets import get_afa_dataset_fn
from afa_rl.shim2018.models import Shim2018MaskedClassifier, Shim2018NNMaskedClassifier
from afa_rl.shim2018.scripts.pretrain_shim2018 import get_shim2018_model_from_config
from afa_rl.utils import check_masked_classifier_performance
from afa_rl.zannone2019.agents import Zannone2019Agent
from afa_rl.zannone2019.models import (
    Zannone2019MaskedClassifier,
    Zannone2019NNMaskedClassifier,
)
from afa_rl.zannone2019.scripts.pretrain_zannone2019 import (
    get_zannone2019_model_from_config,
)
from common.custom_types import (
    AFADataset,
)
from common.utils import dict_to_namespace, get_class_probabilities, set_seed


def get_eval_metrics(eval_tds: list[TensorDictBase]) -> dict[str, Any]:
    eval_metrics = {}
    eval_metrics["reward_sum"] = 0.0
    for td in eval_tds:
        eval_metrics["reward_sum"] += td["next", "reward"].sum()
    eval_metrics["reward_sum"] /= len(eval_tds)
    return eval_metrics


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
):
    set_seed(seed)
    torch.set_float32_matmul_precision("medium")

    # Load train config
    with open(train_config_path, "r") as file:
        train_config_dict: dict = yaml.safe_load(file)
    train_config = dict_to_namespace(train_config_dict)
    device = torch.device(train_config.device)

    # Load pretrain config
    with open(pretrain_config_path, "r") as file:
        pretrain_config_dict: dict = yaml.safe_load(file)
    pretrain_config = dict_to_namespace(pretrain_config_dict)

    # Import is delayed until now to avoid circular imports
    from common.registry import AFA_DATASET_REGISTRY

    train_dataset: AFADataset = AFA_DATASET_REGISTRY[dataset_type].load(
        train_dataset_path
    )
    val_dataset: AFADataset = AFA_DATASET_REGISTRY[dataset_type].load(val_dataset_path)

    # Get the number of features and classes from the dataset
    n_features = train_dataset.features.shape[-1]
    n_classes = train_dataset.labels.shape[-1]

    train_class_probabilities = get_class_probabilities(train_dataset.labels)
    print(f"Class probabilities in training set: {train_class_probabilities}")
    pretrained_model = get_zannone2019_model_from_config(
        pretrain_config, n_features, n_classes, train_class_probabilities
    )
    pretrained_model_checkpoint = torch.load(
        pretrained_model_path / "model.pt", weights_only=True
    )
    pretrained_model.load_state_dict(pretrained_model_checkpoint["state_dict"])
    pretrained_model = pretrained_model.to(device)
    pretrained_model.eval()
    pretrained_model.requires_grad_(False)

    # Check that pretrained model indeed has decent performance
    class_weights = 1 / train_class_probabilities
    class_weights = class_weights / class_weights.sum()
    class_weights_device = class_weights.to(device)
    check_masked_classifier_performance(
        masked_classifier=Zannone2019MaskedClassifier(pretrained_model),
        dataset=val_dataset,
        class_weights=class_weights,
    )

    # The RL reward function depends on a specific AFAClassifier
    reward_fn = get_common_reward_fn(
        Zannone2019MaskedClassifier(pretrained_model),
        loss_fn=partial(F.cross_entropy, weight=class_weights_device),
    )

    # MDP expects special dataset functions
    train_dataset_fn = get_afa_dataset_fn(train_dataset.features, train_dataset.labels)
    val_dataset_fn = get_afa_dataset_fn(val_dataset.features, val_dataset.labels)

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

    eval_env = AFAEnv(
        dataset_fn=val_dataset_fn,
        reward_fn=reward_fn,
        device=device,
        batch_size=torch.Size((1,)),
        feature_size=n_features,
        n_classes=n_classes,
        hard_budget=hard_budget,
    )

    agent: Agent = Zannone2019Agent(
        action_spec=train_env.action_spec,
        _device=device,
        gamma=train_config.agent.gamma,
        lmbda=train_config.agent.lmbda,
        clip_epsilon=train_config.agent.clip_epsilon,
        entropy_bonus=train_config.agent.entropy_bonus,
        entropy_coef=train_config.agent.entropy_coef,
        critic_coef=train_config.agent.critic_coef,
        loss_critic_type=train_config.agent.loss_critic_type,
        lr=train_config.agent.lr,
        max_grad_norm=train_config.agent.max_grad_norm,
        batch_size=train_config.batch_size,
        sub_batch_size=train_config.agent.sub_batch_size,
        num_epochs=train_config.agent.num_epochs,
        replay_buffer_device=device,
        # subclass kwargs
        pointnet=pretrained_model.partial_vae.pointnet,
        encoder=pretrained_model.partial_vae.encoder,
        latent_size=pretrain_config.partial_vae.latent_size,
    )

    collector = SyncDataCollector(
        train_env,
        agent.policy,
        frames_per_batch=train_config.batch_size,
        total_frames=train_config.n_batches * train_config.batch_size,
        # device=device,
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
            loss_info = agent.process_batch(td)

            # Log training info
            run.log(
                {
                    f"train/{k}": v
                    for k, v in (
                        loss_info
                        | agent.get_train_info()
                        | {"reward": td["next", "reward"].mean().item()}
                    ).items()
                },
            )

            if batch_idx != 0 and batch_idx % train_config.eval_every_n_batches == 0:
                with (
                    torch.no_grad(),
                    set_exploration_type(ExplorationType.DETERMINISTIC),
                ):
                    td_evals = [
                        eval_env.rollout(train_config.eval_max_steps, agent.policy)
                        for _ in tqdm(
                            range(train_config.n_eval_episodes), desc="Evaluating"
                        )
                    ]
                metrics_eval = get_eval_metrics(td_evals)
                run.log(
                    {
                        f"eval/{k}": v
                        for k, v in (metrics_eval | agent.get_eval_info()).items()
                    }
                )

    except KeyboardInterrupt:
        pass
    finally:
        run.finish()

        # Check that embedder+classifier still have decent performance
        check_masked_classifier_performance(
            masked_classifier=Zannone2019MaskedClassifier(pretrained_model),
            dataset=val_dataset,
            class_weights=class_weights,
        )

        # Convert the embedder+agent to an AFAMethod and save it
        afa_method = RLAFAMethod(
            agent,
            Zannone2019NNMaskedClassifier(pretrained_model),
        )
        afa_method.save(afa_method_path / "model.pt")

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

        print(f"RLAFAMethod saved to {afa_method_path}")

        # Now load the method
        afa_method = RLAFAMethod.load(
            afa_method_path / "model.pt", device=torch.device("cpu")
        )

        # # Check that the classifier still has decent performance
        check_masked_classifier_performance(
            masked_classifier=afa_method.classifier,
            dataset=val_dataset,
            class_weights=class_weights,
        )


if __name__ == "__main__":
    from common.registry import AFA_DATASET_REGISTRY

    # Use argparse to choose config file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrain_config_path",
        type=Path,
        default="configs/zannone2019/pretrain_zannone2019.yml",
        help="Path to YAML config file used for pretraining",
    )
    parser.add_argument(
        "--train_config_path",
        type=Path,
        default="configs/zannone2019/train_zannone2019.yml",
        help="Path to YAML config file for this training",
    )
    parser.add_argument(
        "--dataset_type", type=str, default="cube", choices=AFA_DATASET_REGISTRY.keys()
    )
    parser.add_argument(
        "--train_dataset_path", type=Path, default="data/cube/train_split_1.pt"
    )
    parser.add_argument(
        "--val_dataset_path", type=Path, default="data/cube/val_split_1.pt"
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=Path,
        default="models/pretrained/zannone2019/temp",
        help="Path to pretrained model folder",
    )
    parser.add_argument("--hard_budget", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--afa_method_path",
        type=Path,
        default="models/afa_methods/zannone2019/temp",
        help="Path to folder to save the trained AFA method",
    )
    args = parser.parse_args()

    main(
        pretrain_config_path=args.pretrain_config_path,
        train_config_path=args.train_config_path,
        dataset_type=args.dataset_type,
        train_dataset_path=args.train_dataset_path,
        val_dataset_path=args.val_dataset_path,
        pretrained_model_path=args.pretrained_model_path,
        hard_budget=args.hard_budget,
        seed=args.seed,
        afa_method_path=args.afa_method_path,
    )
