import argparse
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
import os

import yaml

from afa_rl.afa_methods import AFAContextSmartMethod
from common.registry import AFA_DATASET_REGISTRY, AFA_METHOD_REGISTRY
from coolname import generate_slug


def save_smart_method(hard_budget: int, dataset_train_path: Path, dataset_val_path: Path, seed: int, method_folder: Path):
    # Generate a unique datetime string and create folders
    timestr = generate_slug(2)
    (method_folder / timestr).mkdir(
        parents=True, exist_ok=True
    )

    method = AFAContextSmartMethod()

    method.save(
        method_folder / timestr / "model.pt"
    )
    # Also save a yaml file with params
    with open(
        (method_folder / timestr / "params.yml"), "w"
    ) as f:
        yaml.dump({
            "hard_budget": hard_budget,
            "seed": seed,
            "dataset_type": "AFAContext",
            "train_dataset_path": str(dataset_train_path),
            "val_dataset_path": str(dataset_val_path),
        }, f, default_flow_style=False)



def main(args: argparse.Namespace):

    print("Generating smart AFAContext method...")
    # Create a method for each (train_path, val_path) and seed combination
    for split in range(1, args.n_splits+1):
        train_path = args.dataset_folder / f"train_split_{split}.pt"
        val_path = args.dataset_folder / f"val_split_{split}.pt"

        # The data should exist
        assert train_path.exists() and val_path.exists()

        for seed in range(1, args.n_seeds+1):
            save_smart_method(
                args.hard_budget,
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
        default="models/methods/afacontext_smart",
        help="Path to the method folder",
    )
    parser.add_argument(
        "--dataset_folder",
        type=Path,
        default="data",
        help="Path to the dataset folder",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=3,
        help="Number of splits for the dataset",
    )
    parser.add_argument(
        "--hard_budget",
        type=int,
        default=10,
        help="Hard budget for the method",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=5,
        help="How many different seeds to use",
    )
    args = parser.parse_args()
    main(args)
