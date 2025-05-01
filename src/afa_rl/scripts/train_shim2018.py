import argparse
from functools import partial
from pathlib import Path

import torch
from torch import Tensor
import yaml
from torch import optim
from torch.nn import functional as F
from torchrl.collectors import SyncDataCollector
from torchrl.envs import ExplorationType, StepCounter, TransformedEnv, check_env_specs, set_exploration_type
from tqdm import tqdm

import wandb
from afa_rl.afa_env import AFAEnv, Float, get_common_reward_fn
from afa_rl.afa_methods import Shim2018AFAMethod
from afa_rl.agents import Shim2018Agent
from afa_rl.custom_types import AFAClassifier, Logits
from afa_rl.datasets import get_afa_dataset_fn
from afa_rl.models import ShimEmbedderClassifier
from afa_rl.scripts.pretrain_shim2018 import get_shim2018_model_from_config
from afa_rl.utils import get_sequential_module_norm
from common.custom_types import (
    AFADataset,
    MaskedFeatures,
    FeatureMask,
)
from common.utils import get_class_probabilities, set_seed, dict_to_namespace


def check_embedder_and_classifier(embedder_and_classifier, dataset: AFADataset, class_weights: Float[Tensor, "n_classes"]):
    """
    Check that the embedder and classifier have decent performance on Cube dataset
    """
    # Calculate average accuracy over the whole dataset
    with torch.no_grad():
        # Get the features and labels from the dataset
        features, labels = dataset.get_all_data()
        features = features.to(embedder_and_classifier.device)
        labels = labels.to(embedder_and_classifier.device)

        # Allow embedder to look at *all* features
        masked_features_all = features
        feature_mask_all = torch.ones_like(
            features,
            dtype=torch.bool,
            device=embedder_and_classifier.device,
        )
        embeddings_all = embedder_and_classifier.embedder(
            masked_features_all, feature_mask_all
        )
        predictions_all = embedder_and_classifier.classifier(embeddings_all)
        accuracy_all = (
            (predictions_all.argmax(dim=-1) == labels.argmax(dim=-1)).float().mean()
        )

        # Same thing, but only allow embedder to look at 50% of the features
        feature_mask_half = torch.randint(
            0, 2, feature_mask_all.shape, device=embedder_and_classifier.device
        )
        masked_features_half = features.clone()
        masked_features_half[feature_mask_half == 0] = 0
        embeddings_half = embedder_and_classifier.embedder(
            masked_features_half, feature_mask_half
        )
        predictions_half = embedder_and_classifier.classifier(embeddings_half)
        accuracy_half = (
            (predictions_half.argmax(dim=-1) == labels.argmax(dim=-1)).float().mean()
        )

        # Calculate the loss for the 50% feature case. Useful for setting acquisition costs
        loss_half = F.cross_entropy(predictions_half, labels.float(), weight=class_weights)

        print(
            f"Embedder and classifier accuracy with all features: {accuracy_all.item() * 100:.2f}%"
        )
        print(
            f"Embedder and classifier accuracy with 50% features: {accuracy_half.item() * 100:.2f}%"
        )
        print(f"Average cross-entropy loss with 50% features: {loss_half.item():.4f}")

def train_log(run, agent_loss, td, agent, classifier, embedder, batch_idx):
    run.log(
        {
            "train/agent_loss": agent_loss,
            "train/reward": td["next", "reward"].mean().item(),
            "train/qvalue norm": get_sequential_module_norm(
                agent.value_module.net
            ),
            "train/classifier_norm": get_sequential_module_norm(classifier.mlp),
            "train/embedder_norm": get_sequential_module_norm(
                embedder.encoder.write_block
            ),
            "train/eps": agent.egreedy_module.eps.item(),
            # acquired_features might be NaN if no agent in the batch has a finished episode
            "train/acquired_features": td["feature_mask"][
                td["next", "done"].squeeze(-1)
            ]
            .sum(dim=-1)
            .float()
            .mean()
            .item(),
            "train/replay buffer size": len(agent.replay_buffer),
            "train/batch_idx": batch_idx,
        }
    )

def eval_log(run, eval_metrics, batch_idx):
    run.log(
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
            "eval/reward sum": eval_metrics["reward sum"].mean().item(),
            "eval/traj_len": eval_metrics["traj_len"]
            .float()
            .mean()
            .item(),
            "eval/batch_idx": batch_idx,
        }
    )

def get_eval_metrics(agent, eval_env, train_config, device, classifier, embedder, train_env):
    # HACK: Set the action spec of the agent to the eval env action spec
    # EGreedyModule insists on having the same batch size all the time
    agent.egreedy_module._spec = eval_env.action_spec
    eval_metrics = {
        "acquired_features": torch.zeros(
            train_config.eval_episodes,
            dtype=torch.int64,
            device=device,
        ),
        "is_correct_class": torch.zeros(
            train_config.eval_episodes,
            dtype=torch.bool,
            device=device,
        ),
        "predicted_classes": torch.zeros(
            train_config.eval_episodes,
            dtype=torch.int64,
            device=device,
        ),
        "reward sum": torch.zeros(
            train_config.eval_episodes,
            dtype=torch.float32,
            device=device,
        ),
        "traj_len": torch.zeros(
            train_config.eval_episodes,
            dtype=torch.int64,
            device=device,
        ),
    }
    for eval_episode in tqdm(
        range(train_config.eval_episodes), desc="Evaluating agent..."
    ):
        # Evaluate agent
        td_eval = eval_env.rollout(
            max_steps=9001,
            policy=agent.policy
        )
        # Squeeze batch dimension
        td_eval = td_eval.squeeze(0)
        # Calculate acquired features
        eval_metrics["acquired_features"][eval_episode] = td_eval[
            "feature_mask"
        ][-1].sum()
        # Check whether classification was correct
        logits = classifier(
            embedder(
                td_eval["masked_features"][-1:],
                td_eval["feature_mask"][-1:],
            )
        )
        predicted_class = logits.argmax(dim=-1)
        eval_metrics["predicted_classes"][eval_episode] = (
            predicted_class
        )
        eval_metrics["is_correct_class"][eval_episode] = (
            predicted_class == td_eval["label"][-1].argmax(dim=-1)
        )
        eval_metrics["reward sum"][eval_episode] = td_eval[
            "next", "reward"
        ].sum()
        eval_metrics["traj_len"][eval_episode] = len(td_eval)
    # Reset the action spec of the agent to the train env action spec
    agent.egreedy_module._spec = train_env.action_spec

    return eval_metrics

class Shim2018AFAClassifier(AFAClassifier):
    """
    An adapter for the AFAClassifier interface to be used with the reward function.
    """
    def __init__(self, embedder_and_classifier: ShimEmbedderClassifier):
        # Direct reference to the embedder and classifier
        self.embedder_and_classifier: ShimEmbedderClassifier = embedder_and_classifier

    def __call__(self, masked_features: MaskedFeatures, feature_mask: FeatureMask) -> Logits:
        with torch.no_grad():
            embedding, logits = self.embedder_and_classifier.forward(masked_features, feature_mask)
        return logits


def main(pretrain_config_path: Path, train_config_path: Path, dataset_type: str, train_dataset_path: Path, val_dataset_path: Path, pretrained_model_path: Path, hard_budget: int, seed: int, afa_method_path: Path):
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
    val_dataset: AFADataset = AFA_DATASET_REGISTRY[dataset_type].load(
        val_dataset_path
    )

    # Get the number of features and classes from the dataset
    n_features = train_dataset.features.shape[-1]
    n_classes = train_dataset.labels.shape[-1]

    train_class_probabilities = get_class_probabilities(train_dataset.labels)
    print(f"Class probabilities in training set: {train_class_probabilities}")
    embedder_and_classifier = get_shim2018_model_from_config(pretrain_config, n_features, n_classes, train_class_probabilities)
    embedder_classifier_checkpoint = torch.load(
        pretrained_model_path / "model.pt", weights_only=True
    )
    embedder_and_classifier.load_state_dict(
        embedder_classifier_checkpoint["state_dict"]
    )
    embedder_and_classifier = embedder_and_classifier.to(device)
    embedder = embedder_and_classifier.embedder
    classifier = embedder_and_classifier.classifier

    embedder_classifier_optim = optim.Adam(
        embedder_and_classifier.parameters(), lr=train_config.embedder_classifier.lr
    )

    # Check that embedder+classifier indeed have decent performance
    class_weights = 1 / train_class_probabilities
    class_weights = class_weights / class_weights.sum()
    class_weights_device = class_weights.to(device)
    check_embedder_and_classifier(embedder_and_classifier, val_dataset, class_weights_device)

    # The RL reward function depends on a specific AFAClassifier
    reward_fn = get_common_reward_fn(
        Shim2018AFAClassifier(embedder_and_classifier),
        loss_fn=partial(F.cross_entropy, weight=class_weights_device)
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

    agent = Shim2018Agent(
        embedder=embedder,
        embedding_size=pretrain_config.encoder.output_size,
        action_spec=train_env.action_spec,
        lr=train_config.agent.lr,
        update_tau=train_config.agent.update_tau,
        eps_init=train_config.agent.eps_init,
        eps_end=train_config.agent.eps_end,
        eps_steps=train_config.agent.eps_steps,
        device=device,
        replay_buffer_batch_size=train_config.agent.replay_buffer_batch_size,
        replay_buffer_size=train_config.agent.replay_buffer_size,
        num_optim=train_config.agent.num_optim,
        replay_buffer_alpha=train_config.agent.replay_buffer_alpha,
        replay_buffer_beta=train_config.agent.replay_buffer_beta,
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
            agent.add_to_replay_buffer(td)
            agent_loss = agent.train()
            agent.egreedy_module.step()

            # Train classifier and embedder jointly
            embedder_classifier_optim.zero_grad()
            embedding = embedder(td["masked_features"], td["feature_mask"])
            logits = classifier(embedding)
            class_loss = F.cross_entropy(logits, td["label"], weight=class_weights_device)
            class_loss.mean().backward()
            embedder_classifier_optim.step()

            # Logging
            train_log(run, agent_loss, td, agent, classifier, embedder, batch_idx)

            if batch_idx != 0 and batch_idx % train_config.eval_every_n_batches == 0:
                with (
                    torch.no_grad(),
                    set_exploration_type(ExplorationType.DETERMINISTIC),
                ):
                    eval_metrics = get_eval_metrics(agent, eval_env, train_config, device, classifier, embedder, train_env)
                eval_log(run, eval_metrics, batch_idx)

    except KeyboardInterrupt:
        pass
    finally:

        run.finish()

        # Check that embedder+classifier still have decent performance
        check_embedder_and_classifier(embedder_and_classifier, val_dataset, class_weights_device)

        # Convert the embedder+agent to an AFAMethod and save it
        afa_method = Shim2018AFAMethod(
            device, agent.value_network, embedder_and_classifier,
        )
        afa_method_path.mkdir(parents=True, exist_ok=True)
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

        print(f"Shim2018AFAMethod saved to {afa_method_path}")

        # Now load the method
        afa_method = Shim2018AFAMethod.load(afa_method_path / "model.pt", device)
        # Extract the classifier and embedder and check that they still have decent performance
        check_embedder_and_classifier(afa_method.embedder_and_classifier, val_dataset, class_weights_device)


if __name__ == "__main__":
    from common.registry import AFA_DATASET_REGISTRY
    # Use argparse to choose config file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrain_config_path",
        type=Path,
        required=True,
        help="Path to YAML config file used for pretraining",
    )
    parser.add_argument(
        "--train_config_path",
        type=Path,
        required=True,
        help="Path to YAML config file for this training",
    )
    parser.add_argument("--dataset_type", type=str, required=True, choices=AFA_DATASET_REGISTRY.keys())
    parser.add_argument("--train_dataset_path", type=Path, required=True)
    parser.add_argument("--val_dataset_path", type=Path, required=True)
    parser.add_argument("--pretrained_model_path", type=Path, required=True, help="Path to pretrained model folder")
    parser.add_argument("--hard_budget", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--afa_method_path", type=Path, required=True, help="Path to folder to save the trained AFA method")
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
        afa_method_path=args.afa_method_path
    )
