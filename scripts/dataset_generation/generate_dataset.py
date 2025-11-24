import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import hydra
import torch
from torch.utils.data import random_split

from afabench.common.config_classes import (
    DatasetGenerationConfig,
    SplitRatioConfig,
)
from afabench.common.custom_types import AFADataset
from afabench.common.registry import get_afa_dataset_class

log = logging.getLogger(__name__)


def generate_and_save_split(
    dataset_class: type[AFADataset],
    split_ratio: SplitRatioConfig,
    seed_for_split: int,
    save_path: Path,
    dataset_kwargs: dict[str, Any],
    metadata_to_save: dict[str, Any],
) -> None:
    """
    Generate and save a single train/val/test split.

    Args:
        dataset_class: The dataset class to instantiate.
        split_ratio: The ratio for splitting the dataset into train/val/test.
        seed_for_split: Seed used during splitting.
        save_path: Path to save the generated dataset splits. Will create separate files for each split.
        dataset_kwargs: Keyword arguments to pass to the dataset class constructor.
        metadata_to_save: Additional metadata to save alongside the dataset.
    """
    # Generate full dataset
    dataset = dataset_class(**dataset_kwargs)
    dataset.generate_data()

    # Split into train/val/test
    total_size = len(dataset)
    train_size = int(split_ratio.train * total_size)
    val_size = int(split_ratio.val * total_size)
    test_size = total_size - train_size - val_size

    train_subset, val_subset, test_subset = random_split(
        dataset,  # pyright: ignore[reportArgumentType]
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed_for_split),
    )

    # Create split datasets
    train_dataset = dataset_class(**dataset_kwargs)
    train_dataset.features = dataset.features[train_subset.indices]
    train_dataset.labels = dataset.labels[train_subset.indices]

    val_dataset = dataset_class(**dataset_kwargs)
    val_dataset.features = dataset.features[val_subset.indices]
    val_dataset.labels = dataset.labels[val_subset.indices]

    test_dataset = dataset_class(**dataset_kwargs)
    test_dataset.features = dataset.features[test_subset.indices]
    test_dataset.labels = dataset.labels[test_subset.indices]

    # Save
    save_path.mkdir(parents=True, exist_ok=True)
    train_path = save_path / "train.pt"
    val_path = save_path / "val.pt"
    test_path = save_path / "test.pt"

    train_dataset.save(train_path)
    val_dataset.save(val_path)
    test_dataset.save(test_path)

    # Prepare metadata
    metadata = (
        dataset_kwargs
        | metadata_to_save
        | {
            "seed_for_split": seed_for_split,
            "generated_at": datetime.now(UTC).isoformat(),
        }
    )
    # Save metadata
    with (save_path / "metadata.json").open("w") as f:
        json.dump(metadata, f)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/dataset_generation",
    config_name="config",
)
def main(cfg: DatasetGenerationConfig) -> None:
    for seed in cfg.seeds:
        for instance_idx in cfg.instance_indices:
            dataset_class = get_afa_dataset_class(cfg.dataset.type)
            generate_and_save_split(
                dataset_class=dataset_class,
                split_ratio=cfg.split_ratio,
                seed_for_split=seed,  # use same instance for splitting as for data generation
                save_path=Path(cfg.save_path),
                dataset_kwargs=(
                    dict(cfg.dataset.kwargs)
                    | {"seed": seed, "instance_idx": instance_idx}
                ),
                metadata_to_save={
                    "dataset_type": cfg.dataset.type,
                    "instance_idx": instance_idx,
                },
            )


if __name__ == "__main__":
    main()
