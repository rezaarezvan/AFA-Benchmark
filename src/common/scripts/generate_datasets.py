#!/usr/bin/env python3

import os
import argparse
import torch
import copy
from torch.utils.data import random_split, Subset
from common.registry import AFA_DATASET_REGISTRY

# Define split ratios
SPLIT_RATIOS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

# Base seeds for each split (will be extended based on num_splits)
BASE_SEEDS = [42, 123, 456, 789, 101112, 131415, 161718, 192021, 222324, 252627]

def parse_args():
    parser = argparse.ArgumentParser(description='Generate multiple splits for AFA datasets')
    parser.add_argument('--num_splits', type=int, default=5,
                      help='Number of different splits to generate for each dataset (default: 5)')
    parser.add_argument('--data_dir', type=str, default='data',
                      help='Directory to save the generated datasets (default: data)')
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

def generate_and_save_splits(dataset_class, dataset_name, split_idx, data_dir, **dataset_kwargs):
    """Generate and save train/val/test splits for a dataset with a specific seed."""
    # Set the seed for this split
    seed = BASE_SEEDS[split_idx]
    
    print(f"\nGenerating {dataset_name} dataset (split {split_idx+1}/{args.num_splits}) with seed {seed}...")
    
    # Create dataset with the specific seed
    dataset = dataset_class(**dataset_kwargs)
    dataset.generate_data()
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(SPLIT_RATIOS["train"] * total_size)
    val_size = int(SPLIT_RATIOS["val"] * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_subset, val_subset, test_subset = random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(seed)
    )
    
    # Create new dataset instances for each split
    train_dataset = create_split_dataset(dataset, train_subset, "train", split_idx)
    val_dataset = create_split_dataset(dataset, val_subset, "val", split_idx)
    test_dataset = create_split_dataset(dataset, test_subset, "test", split_idx)
    
    # Create dataset directory
    dataset_dir = os.path.join(data_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Save splits
    train_dataset.save(os.path.join(dataset_dir, f"train_split_{split_idx+1}.pt"))
    val_dataset.save(os.path.join(dataset_dir, f"val_split_{split_idx+1}.pt"))
    test_dataset.save(os.path.join(dataset_dir, f"test_split_{split_idx+1}.pt"))
    
    print(f"Saved {dataset_name} splits to {dataset_dir}")
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset

def generate_all_splits(dataset_class, dataset_name, data_dir, **dataset_kwargs):
    """Generate all splits for a dataset."""
    all_splits = []
    
    for i in range(args.num_splits):
        splits = generate_and_save_splits(dataset_class, dataset_name, i, data_dir, **dataset_kwargs)
        all_splits.append(splits)
    
    return all_splits

def verify_dataset(dataset_name, split_idx, data_dir):
    """Verify that a dataset can be loaded correctly."""
    dataset_class = AFA_DATASET_REGISTRY[dataset_name]
    
    # Load train, val, and test splits
    train_path = os.path.join(data_dir, dataset_name, f"train_split_{split_idx+1}.pt")
    val_path = os.path.join(data_dir, dataset_name, f"val_split_{split_idx+1}.pt")
    test_path = os.path.join(data_dir, dataset_name, f"test_split_{split_idx+1}.pt")
    
    train_dataset = dataset_class.load(train_path)
    val_dataset = dataset_class.load(val_path)
    test_dataset = dataset_class.load(test_path)
    
    print(f"Verified {dataset_name} split {split_idx+1}:")
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
        print(f"Warning: Requested {args.num_splits} splits but only {len(BASE_SEEDS)} seeds available.")
        print("Some splits will reuse seeds.")
    
    
    # Generate splits for all registered datasets
    for dataset_name, dataset_class in AFA_DATASET_REGISTRY.items():
        print(f"\nProcessing dataset: {dataset_name}")
        
        # Set default parameters based on dataset type
        if dataset_name == "cube":
            kwargs = {
                "n_features": 20,
                "n_samples": 1000,
                "informative_feature_variance": 0.2,
                "non_informative_feature_mean": 0.5,
                "non_informative_feature_variance": 0.3
            }
        elif dataset_name == "AFAContext":
            kwargs = {
                "n_samples": 1000,
                "sigma_bin": 0.1,
                "sigma_cube": 0.3,
                "bin_feature_cost": 5.0,
                "n_dummy_features": 10,
                "non_informative_feature_mean": 0.5,
                "non_informative_feature_variance": 0.3
            }
        elif dataset_name == "MNIST":
            kwargs = {
                "train": True,  # Use training set as base
                "transform": None,  # Will use default ToTensor()
                "download": True,
                "root": os.path.join(data_dir, "MNIST")
            }
        elif dataset_name == "diabetes":
            kwargs = {
                "data_path": "datasets/diabetes.csv"
            }
        else:
            print(f"Warning: Unknown dataset type {dataset_name}, skipping...")
            continue
        
        # Generate splits
        splits = generate_all_splits(dataset_class, dataset_name, data_dir, **kwargs)
        
        # Verify first split
        verify_dataset(dataset_name, 0, data_dir)

if __name__ == "__main__":
    main() 