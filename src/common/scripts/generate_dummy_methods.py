"""
Generates .pt files for the two dummy methods SequentialDummyAFAMethod and RandomDummyAFAMethod
"""


import argparse
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
import os

import yaml

from common.registry import AFA_DATASET_REGISTRY, AFA_METHOD_REGISTRY
from coolname import generate_slug


def save_dummy_method(method_name: str, hard_budget: int, dataset_train_path: Path, dataset_val_path: Path, seed: int, method_folder: Path):
    # Find method class using registry
    method_cls = AFA_METHOD_REGISTRY[method_name]

    # Find dataset type from the dataset path
    dataset_type = dataset_train_path.parent.name
    # Find dataset class using registry
    dataset_cls = AFA_DATASET_REGISTRY[dataset_type]

    # Generate a unique datetime string and create folders
    timestr = generate_slug(2)
    (method_folder / method_name / timestr).mkdir(
        parents=True, exist_ok=True
    )

    method = method_cls(
        device=torch.device("cpu"),
        n_classes=dataset_cls.n_classes,
    )

    method.save(
        method_folder / method_name / timestr / "model.pt"
    )
    # Also save a yaml file with params
    with open(
        (method_folder / method_name / timestr / "params.yml"), "w"
    ) as f:
        yaml.dump({
            "hard_budget": hard_budget,
            "seed": seed,
            "dataset_type": dataset_type,
            "train_dataset_path": str(dataset_train_path),
            "val_dataset_path": str(dataset_val_path),
        }, f, default_flow_style=False)



def main(args: argparse.Namespace):
    # The dataset type is the name of the folder containing the dataset
    args.dataset_type = args.dataset_folder.name

    print("Generating dummy methods...")
    # Create a method for each (train_path, val_path) and seed combination
    for split in range(1, args.n_splits+1):
        train_path = args.dataset_folder / f"train_split_{split}.pt"
        val_path = args.dataset_folder / f"val_split_{split}.pt"

        # The data should exist
        assert train_path.exists() and val_path.exists()

        for seed in range(1, args.n_seeds+1):
            for hard_budget in args.hard_budgets:
                save_dummy_method(
                    "sequential_dummy",
                    hard_budget,
                    train_path,
                    val_path,
                    seed,
                    method_folder=args.method_folder,
                )

                save_dummy_method(
                    "random_dummy",
                    hard_budget,
                    train_path,
                    val_path,
                    seed,
                    method_folder=args.method_folder,
                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method_folder",
        type=Path,
        default="models/methods",
        help="Path to the method folder",
    )
    parser.add_argument(
        "--dataset_folder",
        type=Path,
        default=Path("data/cube"),
        help="Which dataset the model is trained on",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=2,
        help="Number of splits for the dataset",
    )
    parser.add_argument(
        "--hard_budgets",
        type=int,
        nargs='+',
        default=[5,10,20],
        help="List of hard budgets for the method",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=2,
        help="How many different seeds to use",
    )
    args = parser.parse_args()
    main(args)
