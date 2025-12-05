"""Generate multiple instances of a dataset, see dataset_generation.md."""

import json
import logging
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import hydra

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
        save_path: Path to save the generated dataset splits. Will create separate folders for each split instance.
        dataset_kwargs: Keyword arguments to pass to the dataset class constructor.
        metadata_to_save: Additional metadata to save alongside the dataset.
    """
    # Generate full dataset
    dataset = dataset_class(**dataset_kwargs)

    # Split into train/val/test
    total_size = len(dataset)
    train_size = int(split_ratio.train * total_size)
    val_size = int(split_ratio.val * total_size)

    all_indices = list(range(total_size))
    rnd = random.Random(seed_for_split)
    rnd.shuffle(all_indices)

    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size : train_size + val_size]
    test_indices = all_indices[train_size + val_size :]

    train_dataset = dataset.create_subset(train_indices)
    val_dataset = dataset.create_subset(val_indices)
    test_dataset = dataset.create_subset(test_indices)

    # Save splits
    save_path.mkdir(parents=True, exist_ok=True)
    train_path = save_path / "train.pt"
    val_path = save_path / "val.pt"
    test_path = save_path / "test.pt"

    for object, path in zip(
        [train_dataset, val_dataset, test_dataset],
        [train_path, val_path, test_path],
    ):
        save_artifact(
            object=object,
            path=path,
            metadata=metadata_to_save
            | {
                "seed_for_split": seed_for_split,
                "generated_at": datetime.now(UTC).isoformat(),
            },
        )

    # Prepare metadata
    metadata_to_save = metadata_to_save | {
        "seed_for_split": seed_for_split,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    json_data = metadata_to_save | {
        "kwargs": dataset_kwargs,
    }
    # Save metadata
    with (save_path / "metadata.json").open("w") as f:
        json.dump(json_data, f)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/dataset_generation",
    config_name="config",
)
def main(cfg: DatasetGenerationConfig) -> None:
    for instance_idx, seed in zip(
        cfg.instance_indices, cfg.seeds, strict=True
    ):
        dataset_class = get_afa_dataset_class(cfg.dataset.class_name)
        if dataset_class.accepts_seed():
            dataset_kwargs = dict(cfg.dataset.kwargs) | {"seed": seed}
        else:
            dataset_kwargs = dict(cfg.dataset.kwargs)
        generate_and_save_split(
            dataset_class=dataset_class,
            split_ratio=cfg.split_ratio,
            # use same instance for splitting as for data generation
            seed_for_split=seed,
            save_path=Path(cfg.save_path) / str(instance_idx),
            dataset_kwargs=dataset_kwargs,
            metadata_to_save={
                "instance_idx": instance_idx,
            },
        )


if __name__ == "__main__":
    main()
