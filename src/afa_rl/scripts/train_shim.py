from argparse import ArgumentParser

import torch
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
from afa_rl.utils import FloatWrapFn


def main():
    # checkpoint as arg
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a ShimEmbedderClassifier checkpoint",
    )
    parser.add_argument(
        "--agent_save_path",
        type=str,
        required=True,
        help="Where to save the trained ShimQAgent",
    )
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, weights_only=True)

    n_features = 20

    embedder_and_classifier = ShimEmbedderClassifier(
        embedder=ShimEmbedder(
            encoder=ReadProcessEncoder(
                feature_size=n_features + 1,  # state contains one value and one index
                output_size=16,
                reading_block_cells=[32, 32],
                writing_block_cells=[32, 32],
                memory_size=16,
                processing_steps=5,
            ),
        ),
        classifier=MLPClassifier(16, 8, [32, 32]),
        lr=1e-4,
    )
    embedder_and_classifier.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda")
    embedder_and_classifier = embedder_and_classifier.to(device)
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
        n_features=n_features, data_points=100_000, sigma=0.01, seed=42
    )
    dataset_fn = get_dataset_fn(dataset.features, dataset.labels)

    batch_size = 1
    env = AFAMDP(
        dataset_fn=dataset_fn,
        embedder=embedder,
        task_model=classifier,
        loss_fn=FloatWrapFn(nn.CrossEntropyLoss(reduction="none")),
        # loss_fn=nn.CrossEntropyLoss(reduction="none"),
        acquisition_costs=(-0.05)
        * torch.ones((n_features,), dtype=torch.float32, device=device)
        / n_features,
        device=device,
        batch_size=torch.Size((batch_size,)),
    )

    agent = ShimQAgent(
        embedding_size=16,
        action_spec=env.action_spec,
        lr=1e-3,
        update_tau=1e-3,
        eps=0.1,
        device=device,
    )

    # Use WandB for logging
    run = wandb.init()

    # Training loop
    try:
        td = env.reset()
        episode_idx = 0
        for i in tqdm(range(100_000)):
            # Pick action
            td = agent.policy(td)
            td = env.step(td)

            agent.optim.zero_grad()

            if not td["action_mask"][:,1:].any().item():
                pass
            agent_loss = agent.loss_module(td)
            agent_loss_value = agent_loss["loss"]
            if agent_loss_value.isnan().any():
                pass
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
                    "agent_loss": agent_loss_value.item(),
                    "action": td["action"].item(),
                    # "classifier_loss": classifier_loss.mean().item(),
                    "fa_reward": td["next", "fa_reward"].sum().item(),
                    "model_reward": td["next","model_reward"].sum().item(),
                    "embedding_norm": td["embedding"].norm().item(),
                    "episode_idx": episode_idx,
                }
            )

            if td["next", "done"].any():
                td = env.reset()
                episode_idx += 1
            else:
                td = td["next"]
    except KeyboardInterrupt:
        pass
    finally:
        agent.save(args.agent_save_path)


if __name__ == "__main__":
    main()
