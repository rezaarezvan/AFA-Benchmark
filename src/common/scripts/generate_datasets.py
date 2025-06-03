#!/usr/bin/env python3

from datetime import datetime
import os
import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import strftime
import torch
import copy
from torch.utils.data import random_split
from wandb.sdk.wandb_run import Run
from common.registry import AFA_DATASET_REGISTRY
import wandb

# Define split ratios
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}

# Base seeds for each split (will be extended based on num_splits)
BASE_SEEDS = [42, 123, 456, 789, 101112, 131415, 161718, 192021, 222324, 252627]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate multiple splits for AFA datasets"
    )
    parser.add_argument(
        "--num_splits",
        type=int,
        default=5,
        help="Number of different splits to generate for each dataset (default: 5)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory to save the generated datasets (default: data)",
    )
    parser.add_argument(
        "--artifact_alias",
        type=str,
        default=strftime("%b%d"),
        help="An alias to add to all generated dataset artifacts.",
    )
    return parser.parse_args()


def create_split_dataset(original_dataset, subset, split_name, split_idx):
    """Create a new dataset instance for a split by copying the original dataset and replacing features/labels."""
    # Create a deep copy of the original dataset
    new_dataset = copy.deepcopy(original_dataset)

    # Get the indices from the subset
    indices = subset.indices

    # Replace features and labels with the subset
    new_dataset.features = original_dataset.features[indices]
    new_dataset.labels = original_dataset.labels[indices]

    return new_dataset


def generate_and_save_splits(
    run: Run,
    dataset_class,
    dataset_type,
    split_idx,
    data_dir,
    artifact_alias: str | None = None,
    **dataset_kwargs,
):
    """Generate and save train/val/test splits for a dataset with a specific seed."""
    # Set the seed for this split
    seed = BASE_SEEDS[split_idx]

    print(
        f"\nGenerating {dataset_type} dataset (split {split_idx + 1}/{args.num_splits}) with seed {seed}..."
    )

    # Create dataset with the specific seed
    dataset = dataset_class(**dataset_kwargs)
    # dataset.generate_data()

    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(SPLIT_RATIOS["train"] * total_size)
    val_size = int(SPLIT_RATIOS["val"] * total_size)
    test_size = total_size - train_size - val_size

    # Split dataset
    train_subset, val_subset, test_subset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    # Create new dataset instances for each split
    train_dataset = create_split_dataset(dataset, train_subset, "train", split_idx)
    val_dataset = create_split_dataset(dataset, val_subset, "val", split_idx)
    test_dataset = create_split_dataset(dataset, test_subset, "test", split_idx)

    if dataset_type in ("miniboone", "physionet"):
        feat = train_dataset.features
        mean = feat.mean(dim=0, keepdim=True)
        std = feat.std(dim=0, unbiased=False, keepdim=True)

        for ds in (train_dataset, val_dataset, test_dataset):
            ds.features = (ds.features - mean) / std

    # Create dataset directory
    dataset_dir = os.path.join(data_dir, dataset_type)
    os.makedirs(dataset_dir, exist_ok=True)

    # Save splits locally
    train_path = os.path.join(dataset_dir, f"train_split_{split_idx + 1}.pt")
    train_dataset.save(train_path)
    val_path = os.path.join(dataset_dir, f"val_split_{split_idx + 1}.pt")
    val_dataset.save(val_path)
    test_path = os.path.join(dataset_dir, f"test_split_{split_idx + 1}.pt")
    test_dataset.save(test_path)

    # Also save as wandb artifact
    artifact = wandb.Artifact(
        name=f"{dataset_type}_split_{split_idx + 1}",
        type="dataset",
        metadata=dataset_kwargs
        | {
            "dataset_type": dataset_type,
            "split_idx": split_idx + 1,
            "seed": seed,
        },
    )

    # Add a dummy file with the current time to ensure a new artifact version is created
    with NamedTemporaryFile("w", delete=False) as f:
        f.write(f"Generated at {strftime('%Y-%m-%d %H:%M:%S')}\n")
        dummy_path = f.name  # Save the name before closing

    artifact.add_file(dummy_path, name="dummy.txt")
    artifact.add_file(train_path, name="train.pt")
    artifact.add_file(val_path, name="val.pt")
    artifact.add_file(test_path, name="test.pt")

    if artifact_alias is not None:
        run.log_artifact(artifact, aliases=[artifact_alias])
    else:
        run.log_artifact(artifact)

    print(f"Saved {dataset_type} splits to {dataset_dir}")
    print(
        f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}"
    )


def generate_all_splits(
    run: Run,
    dataset_class,
    dataset_type,
    data_dir,
    artifact_alias: str | None = None,
    **dataset_kwargs,
):
    """Generate all splits for a dataset."""
    for i in range(args.num_splits):
        generate_and_save_splits(
            run,
            dataset_class,
            dataset_type,
            i,
            data_dir,
            artifact_alias=artifact_alias,
            **dataset_kwargs,
        )


def verify_dataset(dataset_type, split_idx, data_dir):
    """Verify that a dataset can be loaded correctly."""
    dataset_class = AFA_DATASET_REGISTRY[dataset_type]

    # Load train, val, and test splits
    train_path = os.path.join(data_dir, dataset_type, f"train_split_{split_idx + 1}.pt")
    val_path = os.path.join(data_dir, dataset_type, f"val_split_{split_idx + 1}.pt")
    test_path = os.path.join(data_dir, dataset_type, f"test_split_{split_idx + 1}.pt")

    train_dataset = dataset_class.load(Path(train_path))
    val_dataset = dataset_class.load(Path(val_path))
    test_dataset = dataset_class.load(Path(test_path))

    print(f"Verified {dataset_type} split {split_idx + 1}:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")

    # Check a sample from each split
    train_features, train_labels = train_dataset.features, train_dataset.labels
    val_features, val_labels = val_dataset.features, val_dataset.labels
    test_features, test_labels = test_dataset.features, test_dataset.labels

    print(f"train_features: {train_features.shape}")
    print(f"train_labels: {train_labels.shape}")
    print(f"train_labels: {train_labels[:10]}")
    print(f"val_features: {val_features.shape}")
    print(f"val_labels: {val_labels.shape}")
    print(f"val_labels: {val_labels[:10]}")
    print(f"test_features: {test_features.shape}")
    print(f"test_labels: {test_labels.shape}")
    print(f"test_labels: {test_labels[:10]}")

    return train_dataset, val_dataset, test_dataset


def main():
    # Parse command line arguments
    global args
    args = parse_args()

    data_dir = args.data_dir
    # Get the repository root directory (parent of 'src')
    os.makedirs(data_dir, exist_ok=True)
    print(f"Using data directory: {data_dir}")

    # Ensure we have enough seeds
    if args.num_splits > len(BASE_SEEDS):
        print(
            f"Warning: Requested {args.num_splits} splits but only {len(BASE_SEEDS)} seeds available."
        )
        print("Some splits will reuse seeds.")

    # Data will be logged as a wandb artifact
    run = wandb.init(job_type="data_generation")

    # Generate splits for all registered datasets
    for dataset_type, dataset_class in AFA_DATASET_REGISTRY.items():
        print(f"\nProcessing dataset: {dataset_type}")

        # Set default parameters based on dataset type
        if dataset_type == "cube":
            kwargs = {
                "n_samples": 1000,
                "informative_feature_std": 0.3,
                "non_informative_feature_mean": 0.5,
                "non_informative_feature_std": 0.3,
            }
        elif dataset_type == "shim2018cube":
            kwargs = {"n_samples": 10000, "sigma": 0.1}
        elif dataset_type == "AFAContext":
            kwargs = {
                "n_samples": 1000,
                "std_bin": 0.1,
                "std_cube": 0.3,
                "bin_feature_cost": 5.0,
                "non_informative_feature_mean": 0.5,
                "non_informative_feature_std": 0.3,
            }
        elif dataset_type == "MNIST":
            kwargs = {
                "train": True,  # Use training set as base
                "transform": None,  # Will use default ToTensor()
                "download": True,
                "root": os.path.join(data_dir, "MNIST"),
            }
        elif dataset_type == "diabetes":
            kwargs = {"data_path": "datasets/diabetes.csv"}
        elif dataset_type == "physionet":
            kwargs = {"data_path": "datasets/physionet_data.csv"}
        elif dataset_type == "miniboone":
            kwargs = {"data_path": "datasets/miniboone.csv"}
        elif dataset_type == "FashionMNIST":
            kwargs = {
                "train": True,  # Use training set as base
                "transform": None,  # Will use default ToTensor()
                "download": True,
                "root": os.path.join(data_dir, "MNIST"),
            }
        else:
            print(f"Warning: Unknown dataset type {dataset_type}, skipping...")
            continue

        # Generate splits
        generate_all_splits(
            run,
            dataset_class,
            dataset_type,
            data_dir,
            artifact_alias=args.artifact_alias,
            **kwargs,
        )

        # Verify first split
        verify_dataset(dataset_type, 0, data_dir)


if __name__ == "__main__":
    main()
