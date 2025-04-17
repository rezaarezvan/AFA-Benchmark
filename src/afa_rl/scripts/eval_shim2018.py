from argparse import ArgumentParser

import torch
from torch import nn
from torchrl.envs import ExplorationType, set_exploration_type
from tqdm import tqdm

import wandb
from afa_rl.afa_env import Shim2018Env
from afa_rl.agents import Shim2018Agent
from afa_rl.models import (
    ShimMLPClassifier,
    ReadProcessEncoder,
    ShimEmbedder,
    ShimEmbedderClassifier,
)
from afa_rl.utils import FloatWrapFn
from common.datasets import CubeDataset, get_dataset_fn


def main():
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
        help="Path to a .pth file containing a saved ShimQAgent",
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
        classifier=ShimMLPClassifier(16, 8, [32, 32]),
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
    env = Shim2018Env(
        dataset_fn=dataset_fn,
        embedder=embedder,
        classifier=classifier,
        loss_fn=FloatWrapFn(nn.CrossEntropyLoss(reduction="none")),
        # loss_fn=nn.CrossEntropyLoss(reduction="none"),
        acquisition_costs=torch.ones((n_features,), dtype=torch.float32, device=device)
        / n_features,
        device=device,
        batch_size=torch.Size((batch_size,)),
    )

    agent = Shim2018Agent.load(args.agent_save_path, device)

    # Use WandB for logging
    run = wandb.init()

    # Evaluation loop
    try:
        td = env.reset()
        for i in tqdm(range(100_000)):
            # Pick action
            with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
                td = agent.policy(td)
            # Take step
            td = env.step(td)

            # Only log if the episode is finished
            if td["next", "done"].item():
                y = td["next", "label"].argmax(dim=-1).item()
                yhat = classifier(td["next", "embedding"]).argmax(dim=-1).item()
                is_correct = y == yhat
                run.log(
                    {
                        "step": td["next", "feature_mask"].sum().item(),
                        "is_correct": float(is_correct),
                    }
                )
                td = env.reset()
            else:
                td = td["next"]

    except KeyboardInterrupt:
        pass
    finally:
        agent.save("checkpoints/shim_q_agent.pth")


if __name__ == "__main__":
    main()
