import argparse

import torch
import yaml
from torch import nn, optim
from torch.nn import functional as F
from torchrl.collectors import SyncDataCollector
from torchrl.envs import ExplorationType, set_exploration_type
from tqdm import tqdm

import wandb
from afa_rl.afa_env import AFAMDP
from afa_rl.agents import ShimQAgent
from afa_rl.datasets import CubeDataset, get_dataset_fn
from afa_rl.models import (
    MLPClassifier,
    ReadProcessEncoder,
    ShimEmbedder,
    ShimEmbedderClassifier,
)
from afa_rl.utils import FloatWrapFn, dict_to_namespace, get_sequential_module_norm


def check_embedder_and_classifier(embedder_and_classifier, dataset):
    """
    Check that the embedder and classifier have decent performance on Cube dataset
    """
    # Calculate average accuracy over the whole dataset
    with torch.no_grad():
        # Get the features and labels from the dataset
        features_all_features = dataset.features.to(embedder_and_classifier.device)
        labels = dataset.labels.to(embedder_and_classifier.device)

        feature_mask_all_features = torch.ones_like(
            features_all_features,
            dtype=torch.bool,
            device=embedder_and_classifier.device,
        )
        embeddings_all_features = embedder_and_classifier.embedder(
            features_all_features, feature_mask_all_features
        )
        predictions_all_features = embedder_and_classifier.classifier(
            embeddings_all_features
        )
        accuracy_all_features = (
            (predictions_all_features.argmax(dim=-1) == labels.argmax(dim=-1))
            .float()
            .mean()
        )

        feature_mask_half_features = torch.randint(
            0, 2, feature_mask_all_features.shape, device=embedder_and_classifier.device
        )
        features_half_features = features_all_features.clone()
        features_half_features[feature_mask_half_features == 0] = 0
        embeddings_half_features = embedder_and_classifier.embedder(
            features_half_features, feature_mask_half_features
        )
        predictions_half_features = embedder_and_classifier.classifier(
            embeddings_half_features
        )
        accuracy_half_features = (
            (predictions_half_features.argmax(dim=-1) == labels.argmax(dim=-1))
            .float()
            .mean()
        )

        loss_half_features = F.cross_entropy(predictions_half_features, labels.float())

        print(
            f"Embedder and classifier accuracy with all features: {accuracy_all_features.item() * 100:.2f}%"
        )
        print(
            f"Embedder and classifier accuracy with 50% features: {accuracy_half_features.item() * 100:.2f}%"
        )
        print(
            f"Average cross-entropy loss with 50% features: {loss_half_features.item():.4f}"
        )


def main():
    torch.set_float32_matmul_precision("medium")

    # Use argparse to choose config file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    args = parser.parse_args()

    # Load config from yaml file
    with open(args.config, "r") as file:
        config_dict: dict = yaml.safe_load(file)

    wandb.init(
        entity=config_dict["wandb"]["entity"],
        project=config_dict["wandb"]["project"],
        config=config_dict,
    )

    config = dict_to_namespace(config_dict)

    # Load pretrain config
    with open(f"configs/afa_rl/{config.pretrain_shim2018_config}", "r") as file:
        pretrain_config_dict: dict = yaml.safe_load(file)
    pretrain_config = dict_to_namespace(pretrain_config_dict)

    embedder_and_classifier = ShimEmbedderClassifier(
        embedder=ShimEmbedder(
            encoder=ReadProcessEncoder(
                feature_size=pretrain_config.n_features
                + 1,  # state contains one value and one index
                output_size=pretrain_config.encoder.output_size,
                reading_block_cells=pretrain_config.encoder.reading_block_cells,
                writing_block_cells=pretrain_config.encoder.writing_block_cells,
                memory_size=pretrain_config.encoder.memory_size,
                processing_steps=pretrain_config.encoder.processing_steps,
            ),
        ),
        classifier=MLPClassifier(
            pretrain_config.encoder.output_size, 8, pretrain_config.classifier.num_cells
        ),
        lr=pretrain_config.embedder_classifier.lr,
    )
    embedder_classifier_checkpoint = torch.load(
        f"checkpoints/afa_rl/{pretrain_config.checkpoint}", weights_only=True
    )
    embedder_and_classifier.load_state_dict(
        embedder_classifier_checkpoint["state_dict"]
    )
    embedder_and_classifier = embedder_and_classifier.to(config.device)
    # embedder_and_classifier.eval()
    embedder = embedder_and_classifier.embedder
    classifier = embedder_and_classifier.classifier

    # Freeze embedder weights
    # embedder.requires_grad_(False)
    # embedder.eval()
    # Freeze classifier weights
    # classifier.requires_grad_(False)
    # classifier.eval()

    embedder_classifier_optim = optim.Adam(
        embedder_and_classifier.parameters(), lr=config.embedder_classifier.lr
    )

    # Prepare cube dataset for the format that AFAMDP expects
    dataset = CubeDataset(
        n_features=pretrain_config.n_features,
        data_points=config.dataset.size,
        sigma=config.dataset.sigma,
        seed=config.dataset.seed,
    )
    dataset_fn = get_dataset_fn(dataset.features, dataset.labels)

    # Check that embedder+classifier indeed have decent performance
    check_embedder_and_classifier(embedder_and_classifier, dataset)

    train_env = AFAMDP(
        dataset_fn=dataset_fn,
        embedder=embedder,
        classifier=classifier,
        loss_fn=FloatWrapFn(nn.CrossEntropyLoss(reduction="none")),
        acquisition_costs=config.mdp.acquisition_cost
        * torch.ones(
            (pretrain_config.n_features,), dtype=torch.float32, device=config.device
        ),
        invalid_action_cost=config.mdp.invalid_action_cost,
        device=config.device,
        batch_size=torch.Size((config.n_agents,)),
    )
    # check_env_specs(train_env)

    eval_env = AFAMDP(
        dataset_fn=dataset_fn,
        embedder=embedder,
        classifier=classifier,
        loss_fn=FloatWrapFn(nn.CrossEntropyLoss(reduction="none")),
        acquisition_costs=config.mdp.acquisition_cost
        * torch.ones(
            (pretrain_config.n_features,), dtype=torch.float32, device=config.device
        )
        / pretrain_config.n_features,
        invalid_action_cost=config.mdp.invalid_action_cost,
        device=config.device,
        batch_size=torch.Size((1,)),
    )

    agent = ShimQAgent(
        embedding_size=pretrain_config.encoder.output_size,
        action_spec=train_env.action_spec,
        lr=config.agent.lr,
        update_tau=config.agent.update_tau,
        eps_init=config.agent.eps_init,
        eps_end=config.agent.eps_end,
        eps_steps=config.agent.eps_steps,
        device=config.device,
        replay_buffer_batch_size=config.agent.replay_buffer_batch_size,
        replay_buffer_size=config.agent.replay_buffer_size,
        num_optim=config.agent.num_optim,
        replay_buffer_alpha=config.agent.replay_buffer_alpha,
        replay_buffer_beta=config.agent.replay_buffer_beta,
    )

    collector = SyncDataCollector(
        train_env,
        agent.policy,
        frames_per_batch=config.batch_size,
        total_frames=config.n_batches * config.batch_size,
    )

    # Use WandB for logging
    run = wandb.init(
        entity=config.wandb.entity,
        project=config.wandb.project,
    )

    # Training loop
    try:
        for batch_idx, td in tqdm(
            enumerate(collector), total=config.n_batches, desc="Training agent..."
        ):
            # Collapse agent and batch dimensions
            td = td.flatten(start_dim=0, end_dim=1)
            agent.add_to_replay_buffer(td)
            agent_loss = agent.train()
            agent.egreedy_module.step()

            # Train classifier and embedder jointly
            embedder_classifier_optim.zero_grad()
            embedding = embedder(td["feature_values"], td["feature_mask"])
            logits = classifier(embedding)
            class_loss = F.cross_entropy(
                logits, td["label"].float()
            ).mean()
            class_loss.mean().backward()
            embedder_classifier_optim.step()

            # Logging
            run.log(
                {
                    "train/agent_loss": agent_loss,
                    # "train/action": td["action"].item(),
                    "train/fa_reward": td["next", "fa_reward"].mean().item(),
                    "train/model_reward": td["next", "model_reward"].mean().item(),
                    "train/invalid_action_reward": td["next", "invalid_action_reward"]
                    .mean()
                    .item(),
                    "train/reward": td["next", "reward"].mean().item(),
                    # "train/embedding_norm": td["embedding"].norm().item(),
                    # "train/episode_idx": episode_idx,
                    "train/qvalue norm": get_sequential_module_norm(
                        agent.value_module.net
                    ),
                    "train/classifier_norm": get_sequential_module_norm(classifier.mlp),
                    "train/embedder_norm~": get_sequential_module_norm(
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
                    "train/accuracy": td["label"][td["next", "done"].squeeze(-1)]
                    .argmax(dim=-1)
                    .eq(td["next", "predicted_class"][td["next", "done"].squeeze(-1)])
                    .float()
                    .mean()
                    .item(),
                    "train/replay buffer size": len(agent.replay_buffer),
                    "train/batch_idx": batch_idx,
                }
            )

            if batch_idx % config.eval_every_n_batches == 0:
                with (
                    torch.no_grad(),
                    set_exploration_type(ExplorationType.DETERMINISTIC),
                ):
                    # HACK: Set the action spec of the agent to the eval env action spec
                    # EGreedyModule insists on having the same batch size all the time
                    agent.egreedy_module._spec = eval_env.action_spec
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
                        range(config.eval_episodes), desc="Evaluating agent..."
                    ):
                        # Evaluate agent
                        td_eval = eval_env.rollout(
                            max_steps=config.eval_max_steps, policy=agent.policy
                        )
                        # Squeeze batch dimension
                        td_eval = td_eval.squeeze(0)
                        # Calculate acquired features
                        eval_metrics["acquired_features"][eval_episode] = td_eval[
                            "feature_mask"
                        ][-1].sum()
                        # Check whether classification was correct
                        predicted_class = classifier(td_eval["embedding"][-1]).argmax(
                            dim=-1
                        )
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
                    # Reset the action spec of the agent to the train env action spec
                    agent.egreedy_module._spec = train_env.action_spec
                    # eval_mean_metrics = {f"eval/{k}": v.float().mean() for k, v in eval_metrics.items()}
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
    # finally:
    #     agent.save(f"checkpoints/afa_rl/{config.checkpoint}")
    #     print(f"Agent saved to checkpoints/afa_rl/{config.checkpoint}")


if __name__ == "__main__":
    main()
