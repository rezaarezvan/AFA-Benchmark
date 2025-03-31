import argparse

import torch
from torchrl.envs import ExplorationType, set_exploration_type
import yaml
from torch import nn
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


def main():
    torch.set_float32_matmul_precision("medium")

    # Use argparse to choose config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
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
    embedder = embedder_and_classifier.embedder
    # Freeze embedder weights
    embedder.requires_grad_(False)
    embedder.eval()
    # Freeze classifier weights
    classifier = embedder_and_classifier.classifier
    classifier.requires_grad_(False)
    classifier.eval()

    # Prepare cube dataset for the format that AFAMDP expects
    dataset = CubeDataset(
        n_features=pretrain_config.n_features,
        data_points=config.dataset.size,
        sigma=config.dataset.sigma,
        seed=config.dataset.seed,
    )
    dataset_fn = get_dataset_fn(dataset.features, dataset.labels)

    batch_size = 1
    env = AFAMDP(
        dataset_fn=dataset_fn,
        embedder=embedder,
        task_model=classifier,
        loss_fn=FloatWrapFn(nn.CrossEntropyLoss(reduction="none")),
        acquisition_costs=config.mdp.acquisition_cost
        * torch.ones(
            (pretrain_config.n_features,), dtype=torch.float32, device=config.device
        )
        / pretrain_config.n_features,
        device=config.device,
        batch_size=torch.Size((batch_size,)),
    )

    agent = ShimQAgent(
        embedding_size=pretrain_config.encoder.output_size,
        action_spec=env.action_spec,
        lr=config.agent.lr,
        update_tau=config.agent.update_tau,
        eps_init=config.agent.eps_init,
        eps_end=config.agent.eps_end,
        eps_steps=config.agent.eps_episodes,
        device=config.device,
    )

    # Use WandB for logging
    run = wandb.init(
        entity=config.wandb.entity,
        project=config.wandb.project,
    )

    # Training loop
    try:
        td = env.reset()
        episode_idx = 0
        pbar = tqdm(total=config.n_episodes)
        while True:
            # Pick action
            td = agent.policy(td)
            td = env.step(td)

            agent.optim.zero_grad()

            agent_loss = agent.loss_module(td)
            agent_loss_value = agent_loss["loss"]
            agent_loss_value.backward()

            # Clip gradients?

            agent.optim.step()

            # Update target network
            agent.updater.step()

            # Train classifier on *all* samples in batch?
            # classifier_optim.zero_grad()
            # classifier_loss = classifier_loss_fn(
            #     classifier(td_maybe_done["embedding"]), td_maybe_done["label"]
            # )
            # classifier_loss.mean().backward()
            # classifier_optim.step()

            # Logging
            run.log(
                {
                    "train/agent_loss": agent_loss_value.item(),
                    "train/action": td["action"].item(),
                    "train/fa_reward": td["next", "fa_reward"].sum().item(),
                    "train/model_reward": td["next", "model_reward"].sum().item(),
                    "train/embedding_norm": td["embedding"].norm().item(),
                    "train/episode_idx": episode_idx,
                    "train/qvalue norm": get_sequential_module_norm(
                        agent.value_module.net
                    ),
                    "train/eps": agent.egreedy_module.eps.item(),
                }
            )

            if td["next", "done"].any():
                td = env.reset()
                agent.egreedy_module.step()
                episode_idx += 1
                pbar.update(1)
                if episode_idx % config.eval_every_n_episodes == 0:
                    with (
                        torch.no_grad(),
                        set_exploration_type(ExplorationType.DETERMINISTIC),
                    ):
                        acquired_features = torch.zeros(
                            config.eval_episodes,
                            dtype=torch.int64,
                            device=config.device,
                        )
                        is_correct_class = torch.zeros(
                            config.eval_episodes, dtype=torch.bool, device=config.device
                        )
                        predicted_classes = torch.zeros(
                            config.eval_episodes,
                            dtype=torch.int64,
                            device=config.device,
                        )
                        print("Evaluating agent...")
                        for eval_episode in range(config.eval_episodes):
                            # Evaluate agent
                            td_eval = env.reset()
                            td_eval = agent.policy(td_eval)
                            td_eval = env.step(td_eval)
                            while not td_eval["next", "done"]:
                                td_eval = td_eval["next"]
                                td_eval = agent.policy(td_eval)
                                td_eval = env.step(td_eval)
                            # Calculate acquired features
                            acquired_features[eval_episode] = td_eval[
                                "feature_mask"
                            ].sum()
                            # Check whether classification was correct
                            predicted_class = classifier(td_eval["embedding"]).argmax(
                                dim=-1
                            )
                            predicted_classes[eval_episode] = predicted_class
                            is_correct_class[eval_episode] = (
                                predicted_class == td_eval["label"].all()
                            )
                        wandb.log(
                            {
                                "eval/acquired_features": acquired_features.float()
                                .mean()
                                .item(),
                                "eval/accuracy": is_correct_class.float().mean().item(),
                                "eval/predicted_class": wandb.Histogram(
                                    predicted_classes.cpu().numpy(),
                                ),
                                "eval/episode_idx": episode_idx,
                            }
                        )
            else:
                td = td["next"]
    except KeyboardInterrupt:
        pass
    finally:
        agent.save(f"checkpoints/afa_rl/{config.checkpoint}")
        print(f"Agent saved to checkpoints/afa_rl/{config.checkpoint}")


if __name__ == "__main__":
    main()
