import argparse
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import nn
import yaml
from jaxtyping import Float
from tensordict import TensorDictBase
from torch import Tensor, optim
from torch.nn import functional as F
from torchrl.collectors import SyncDataCollector
from torchrl.envs import ExplorationType, check_env_specs, set_exploration_type
from afa_rl.agents import Agent
from tqdm import tqdm

import wandb
from afa_rl.afa_env import AFAEnv, get_common_reward_fn
from afa_rl.afa_methods import RLAFAMethod
from afa_rl.shim2018.agents import Shim2018Agent
from afa_rl.custom_types import MaskedClassifier
from afa_rl.datasets import get_afa_dataset_fn
from afa_rl.shim2018.models import (
    LitShim2018EmbedderClassifier,
    Shim2018MaskedClassifier,
    Shim2018NNMaskedClassifier,
)
from afa_rl.shim2018.scripts.pretrain_shim2018 import get_shim2018_model_from_config
from afa_rl.utils import (
    afacontext_optimal_selection,
    check_masked_classifier_performance,
    module_norm,
)
from common.custom_types import (
    AFADataset,
)
from common.models import LitMaskedClassifier, MaskedMLPClassifier
from common.utils import dict_to_namespace, get_class_probabilities, set_seed


def get_eval_metrics(
    eval_tds: list[TensorDictBase], masked_classifier: MaskedClassifier
) -> dict[str, Any]:
    eval_metrics = {}
    eval_metrics["reward_sum"] = 0.0
    eval_metrics["actions"] = []
    n_correct_samples = 0
    for td in eval_tds:
        td = td.cpu()
        eval_metrics["reward_sum"] += td["next", "reward"].sum()
        eval_metrics["actions"].extend(td["action"].tolist())
        # Check whether prediction is correct
        td_end = td[-1:]
        logits = masked_classifier(
            td_end["next", "masked_features"], td_end["next", "feature_mask"]
        )
        if logits.argmax(dim=-1) == td_end["label"].argmax(dim=-1):
            n_correct_samples += 1
    eval_metrics["reward_sum"] /= len(eval_tds)
    eval_metrics["actions"] = wandb.Histogram(eval_metrics["actions"], num_bins=20)
    eval_metrics["accuracy"] = n_correct_samples / len(eval_tds)
    return eval_metrics


def afacontext_benchmark_policy(
    tensordict: TensorDictBase,
) -> TensorDictBase:
    """Select features optimally in AFAContext."""
    masked_features = tensordict["masked_features"]
    feature_mask = tensordict["feature_mask"]

    selection = afacontext_optimal_selection(masked_features, feature_mask)

    tensordict["action"] = selection
    return tensordict


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
    with open(train_config_path) as file:
        train_config_dict: dict = yaml.safe_load(file)
    train_config = dict_to_namespace(train_config_dict)
    device = torch.device(train_config.device)

    # Load pretrain config
    with open(pretrain_config_path) as file:
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
    embedder_and_classifier = get_shim2018_model_from_config(
        pretrain_config, n_features, n_classes, train_class_probabilities
    )
    embedder_classifier_checkpoint = torch.load(
        pretrained_model_path / "model.pt", weights_only=True
    )
    embedder_and_classifier.load_state_dict(
        embedder_classifier_checkpoint["state_dict"]
    )
    embedder_and_classifier = embedder_and_classifier.to(device)
    # embedder = embedder_and_classifier.embedder
    # classifier = embedder_and_classifier.classifier
    # pretrained_model = LitMaskedClassifier.load_from_checkpoint(
    #     pretrained_model_path / "model.pt",
    #     map_location=device,
    #     strict=False,
    #     classifier=MaskedMLPClassifier(
    #         n_features=n_features,
    #         n_classes=n_classes,
    #     )
    # )
    # masked_classifier = pretrained_model.classifier
    # masked_classifier = masked_classifier.to(device)

    embedder_and_classifier_optim = optim.Adam(
        embedder_and_classifier.parameters(), lr=train_config.pretrained_model.lr
    )

    # Check that the pretrained model indeed has decent performance
    class_weights = 1 / train_class_probabilities
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)
    check_masked_classifier_performance(
        masked_classifier=Shim2018MaskedClassifier(embedder_and_classifier),
        dataset=val_dataset,
        class_weights=class_weights,
    )

    # The RL reward function depends on a specific AFAClassifier
    reward_fn = get_common_reward_fn(
        Shim2018NNMaskedClassifier(embedder_and_classifier),
        loss_fn=partial(F.cross_entropy, weight=class_weights),
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

    agent: Agent = Shim2018Agent(
        embedder=embedder_and_classifier.embedder,
        embedding_size=pretrain_config.encoder.output_size,
        n_features=n_features,
        action_mask_key="action_mask",
        action_spec=train_env.action_spec,
        _device=device,
        gamma=1.0,
        loss_function="l2",
        delay_value=train_config.agent.delay_value,
        double_dqn=train_config.agent.double_dqn,
        eps_annealing_num_batches=train_config.agent.eps_steps,
        eps_init=train_config.agent.eps_init,
        eps_end=train_config.agent.eps_end,
        update_tau=train_config.agent.update_tau,
        lr=train_config.agent.lr,
        max_grad_norm=train_config.agent.max_grad_norm,
        replay_buffer_size=train_config.agent.replay_buffer_size,
        sub_batch_size=train_config.agent.replay_buffer_batch_size,
        num_samples=train_config.agent.num_optim,
        replay_buffer_device=device,
        replay_buffer_alpha=train_config.agent.replay_buffer_alpha,
        replay_buffer_beta_init=train_config.agent.replay_buffer_beta_init,
        replay_buffer_beta_end=train_config.agent.replay_buffer_beta_end,
        replay_buffer_beta_annealing_num_batches=train_config.n_batches,
        init_random_frames=train_config.agent.init_random_frames,
    )

    # Manual debugging
    td = train_env.reset()
    td = agent.policy(td)
    td = train_env.step(td)

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

    # TEMP: check how large the loss difference is between good and bad policies
    # features, label = train_dataset[42]
    # bad_feature_mask = torch.zeros_like(features)
    # bad_feature_mask[-4:] = 1
    # bad_masked_features = features * bad_feature_mask
    # bad_logits = masked_classifier(bad_masked_features.unsqueeze(0).to(device), bad_feature_mask.unsqueeze(0).to(device))
    # bad_loss = F.cross_entropy(bad_logits.cpu(), label.unsqueeze(0))
    # print(f"Bad loss: {bad_loss.item()}")

    # good_feature_mask = torch.zeros_like(features)
    # good_feature_mask[0] = 1
    # context = features[0].int().item()
    # good_feature_mask[1+context*3:4+context*3] = 1
    # good_masked_features = features * good_feature_mask
    # good_logits = masked_classifier(good_masked_features.unsqueeze(0).to(device), good_feature_mask.unsqueeze(0).to(device))
    # good_loss = F.cross_entropy(good_logits.cpu(), label.unsqueeze(0))
    # print(f"Good loss: {good_loss.item()}")

    # Training loop
    try:
        for batch_idx, td in tqdm(
            enumerate(collector), total=train_config.n_batches, desc="Training agent..."
        ):
            # Collapse agent and batch dimensions
            td = td.flatten(start_dim=0, end_dim=1)
            loss_info = agent.process_batch(td)

            # Train classifier and embedder jointly if we have reached the correct batch
            if batch_idx >= train_config.activate_joint_training_after_n_batches:
                embedder_and_classifier_optim.zero_grad()
                embedding, logits = embedder_and_classifier(
                    td["masked_features"], td["feature_mask"]
                )
                class_loss = F.cross_entropy(logits, td["label"], weight=class_weights)
                class_loss.mean().backward()
                embedder_and_classifier_optim.step()

            # Log training info
            run.log(
                {
                    f"train/{k}": v
                    for k, v in (
                        loss_info
                        | agent.get_train_info()
                        | {
                            "reward": td["next", "reward"].mean().item(),
                            # "action value": td["action_value"].mean().item(),
                            "chosen action value": td["chosen_action_value"]
                            .mean()
                            .item(),
                            # "actions": wandb.Histogram(
                            #     td["action"].tolist(), num_bins=20
                            # ),
                        }
                    ).items()
                },
            )

            if batch_idx != 0 and batch_idx % train_config.eval_every_n_batches == 0:
                with (
                    torch.no_grad(),
                    set_exploration_type(ExplorationType.DETERMINISTIC),
                ):
                    if batch_idx == 10000:  # TEMP
                        pass
                    # HACK: Set the action spec of the agent to the eval env action spec
                    agent.egreedy_module._spec = eval_env.action_spec
                    td_evals = [
                        eval_env.rollout(
                            train_config.eval_max_steps, agent.policy
                        ).squeeze(0)
                        for _ in tqdm(
                            range(train_config.n_eval_episodes), desc="Evaluating"
                        )
                    ]
                    benchmark_td_evals = [
                        eval_env.rollout(
                            train_config.eval_max_steps, afacontext_benchmark_policy
                        ).squeeze(0)
                        for _ in tqdm(
                            range(train_config.n_eval_episodes), desc="Evaluating"
                        )
                    ]
                    # Reset the action spec of the agent to the train env action spec
                    agent.egreedy_module._spec = train_env.action_spec
                metrics_eval = get_eval_metrics(
                    td_evals, Shim2018MaskedClassifier(embedder_and_classifier)
                )
                benchmark_metrics_eval = get_eval_metrics(
                    benchmark_td_evals,
                    Shim2018MaskedClassifier(embedder_and_classifier),
                )
                run.log(
                    {
                        **{
                            f"eval/{k}": v
                            for k, v in (
                                metrics_eval
                                | agent.get_eval_info()
                                | {
                                    "classifier_norm": module_norm(
                                        embedder_and_classifier.classifier
                                    ),
                                    "embedder_norm": module_norm(
                                        embedder_and_classifier.embedder
                                    ),
                                }
                            ).items()
                        },
                        **{
                            f"benchmark_eval/{k}": v
                            for k, v in benchmark_metrics_eval.items()
                        },
                    }
                )

    except KeyboardInterrupt:
        pass
    finally:
        run.finish()

        # Check performance in the end, after joint training
        check_masked_classifier_performance(
            masked_classifier=Shim2018MaskedClassifier(embedder_and_classifier),
            dataset=val_dataset,
            class_weights=class_weights,
        )
        # HACK: Set the action spec of the agent to the eval env action spec
        agent.egreedy_module._spec = eval_env.action_spec
        with torch.no_grad():
            td_evals = [
                eval_env.rollout(train_config.eval_max_steps, agent.policy).squeeze(0)
                for _ in tqdm(range(train_config.n_eval_episodes), desc="Evaluating")
            ]
        # Reset the action spec of the agent to the train env action spec
        agent.egreedy_module._spec = train_env.action_spec
        metrics_eval = get_eval_metrics(td_evals, Shim2018MaskedClassifier(embedder_and_classifier))

        # Convert the embedder+agent to an AFAMethod and save it
        afa_method = RLAFAMethod(
            agent,
            Shim2018NNMaskedClassifier(embedder_and_classifier),
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


if __name__ == "__main__":
    from common.registry import AFA_DATASET_REGISTRY

    # Use argparse to choose config file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrain_config_path",
        type=Path,
        default="configs/shim2018/pretrain_shim2018.yml",
        help="Path to YAML config file used for pretraining",
    )
    parser.add_argument(
        "--train_config_path",
        type=Path,
        default="configs/shim2018/train_shim2018.yml",
        help="Path to YAML config file for this training",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="AFAContext",
        choices=AFA_DATASET_REGISTRY.keys(),
    )
    parser.add_argument(
        "--train_dataset_path", type=Path, default="data/AFAContext/train_split_1.pt"
    )
    parser.add_argument(
        "--val_dataset_path", type=Path, default="data/AFAContext/val_split_1.pt"
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=Path,
        default="models/pretrained/shim2018/temp",
        help="Path to pretrained model folder",
    )
    parser.add_argument("--hard_budget", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--afa_method_path",
        type=Path,
        default="models/methods/shim2018/temp",
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
