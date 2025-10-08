import copy
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import strftime

import hydra
import torch
from torch.utils.data import random_split

import wandb
from common.config_classes import DatasetGenerationConfig, SplitRatioConfig
from common.custom_types import AFADataset
from common.registry import get_afa_dataset_class
from wandb.sdk.wandb_run import Run


def create_split_dataset(original_dataset, subset):
    """Create a new dataset instance for a split by copying the original dataset and replacing features/labels."""
    # Create a deep copy of the original dataset
    new_dataset = copy.deepcopy(original_dataset)

    # Get the indices from the subset
    indices = subset.indices

    # Replace features and labels with the subset
    new_dataset.features = original_dataset.features[indices]
    new_dataset.labels = original_dataset.labels[indices]
    new_dataset.indices = original_dataset.indices[indices]

    return new_dataset


def generate_and_save_split(
    run: Run,
    dataset_class: type[AFADataset],
    dataset_type: str,
    split_idx: int,
    split_ratio: SplitRatioConfig,
    seed: int,
    data_dir: Path,
    output_artifact_aliases: tuple[str, ...] = (),
    epsilon: float = 1e-8,  # added when dividing by standard deviation to avoid division by zero
    **dataset_kwargs,
):
    """Generate and save a single train/val/test split for a dataset with a specific seed. The seed affects both data generation and split."""
    # Create dataset with the specific seed
    dataset_kwargs["seed"] = seed
    dataset = dataset_class(**dataset_kwargs)
    dataset.generate_data()

    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(split_ratio.train * total_size)
    val_size = int(split_ratio.val * total_size)
    test_size = total_size - train_size - val_size

    # Split dataset
    train_subset, val_subset, test_subset = random_split(
        dataset,  # pyright: ignore[reportArgumentType]
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    # Create new dataset instances for each split
    train_dataset = create_split_dataset(dataset, train_subset)
    val_dataset = create_split_dataset(dataset, val_subset)
    test_dataset = create_split_dataset(dataset, test_subset)

    if dataset_type in ("miniboone", "physionet"):
        feat = train_dataset.features
        mean = feat.mean(dim=0, keepdim=True)
        std = feat.std(dim=0, unbiased=False, keepdim=True)

        for ds in (train_dataset, val_dataset, test_dataset):
            ds.features = (ds.features - mean) / (std + epsilon)

    # Create dataset directory
    dataset_dir = data_dir / dataset_type
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Save splits locally
    train_path = dataset_dir / f"train_split_{split_idx}.pt"
    train_dataset.save(train_path)
    val_path = dataset_dir / f"val_split_{split_idx}.pt"
    val_dataset.save(val_path)
    test_path = dataset_dir / f"test_split_{split_idx}.pt"
    test_dataset.save(test_path)

    # Also save as wandb artifact
    artifact = wandb.Artifact(
        name=f"{dataset_type}_split_{split_idx}",
        type="dataset",
        metadata=dataset_kwargs
        | {
            "dataset_type": dataset_type,
            "split_idx": split_idx,
            "seed": seed,
        },
    )

    # Add a dummy file with the current time to ensure a new artifact version is created
    with NamedTemporaryFile("w", delete=False) as f:
        f.write(f"Generated at {strftime('%Y-%m-%d %H:%M:%S')}\n")
        dummy_path = f.name  # Save the name before closing

    artifact.add_file(dummy_path, name="dummy.txt")
    artifact.add_file(str(train_path), name="train.pt")
    artifact.add_file(str(val_path), name="val.pt")
    artifact.add_file(str(test_path), name="test.pt")

    run.log_artifact(artifact, aliases=list(output_artifact_aliases))

    log.info(f"Saved {dataset_type} split to {dataset_dir}")
    log.info(
        f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}"
    )


log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../conf/dataset_generation",
    config_name="config",
)
def main(cfg: DatasetGenerationConfig) -> None:
    os.makedirs(cfg.data_dir, exist_ok=True)

    # Data will be logged as a wandb artifact
    # Since we often generate data with multiruns, only create a new run if not already running
    run = wandb.run or wandb.init(job_type="data_generation", dir="wandb")

    dataset_class = get_afa_dataset_class(cfg.dataset.type)

    generate_and_save_split(
        run=run,
        dataset_class=dataset_class,
        dataset_type=cfg.dataset.type,
        split_idx=cfg.split_idx,
        split_ratio=cfg.split_ratio,
        seed=cfg.seeds[cfg.split_idx - 1],
        data_dir=Path(cfg.data_dir),
        output_artifact_aliases=tuple(cfg.output_artifact_aliases),
        epsilon=cfg.epsilon,
        **cfg.dataset.kwargs,
    )


if __name__ == "__main__":
    main()
