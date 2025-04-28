"""
Generates .pt files for the two dummy methods SequentialDummyAFAMethod and RandomDummyAFAMethod
"""


import argparse
from datetime import datetime
from pathlib import Path
import torch
from tqdm import tqdm
import os

from afa_rl.afa_methods import RandomDummyAFAMethod, SequentialDummyAFAMethod
from common.registry import AFA_DATASET_REGISTRY, AFA_METHOD_REGISTRY


def save_dummy_method(method_name, hard_budget, dataset_train_path, dataset_val_path):
    # Create folders if necessary
    # (args.models_folder / method_name).mkdir(parents=True, exist_ok=True)

    # Find method class using registry
    method_cls = AFA_METHOD_REGISTRY[method_name]

    # Find dataset type from the dataset path
    dataset_type = dataset_train_path.parent.name
    # Find dataset class using registry
    dataset_cls = AFA_DATASET_REGISTRY[dataset_type]

    # Create five methods
    for i in tqdm(range(1, 6)):
        # Generate a unique datetime string and create folders
        timestr = datetime.now().strftime("%Y%m%d_%H%M%S")
        (args.models_folder / method_name / timestr).mkdir(
            parents=True, exist_ok=True
        )

        method = method_cls(
            device=torch.device("cpu"),
            n_classes=dataset_cls.n_classes,
        )
        method.save(
            (args.models_folder / method_name / timestr / "model.pt")
        )
        # Also save a yaml file with params
        with open(
            (args.models_folder / method_name / timestr / "params.yml"), "w"
        ) as f:
            f.write(f"hard_budget: {hard_budget}\n")
            f.write(f"train_dataset_path: {dataset_train_path}\n")
            f.write(f"val_dataset_path: {dataset_val_path}\n")



def main(args: argparse.Namespace):
    # The dataset type is the parent folder of dataset_train_path
    args.dataset_type = args.dataset_train_path.parent.name
    # Training and validation datasets should have the same type
    assert args.dataset_type == args.dataset_val_path.parent.name

    print("Generating dummy methods...")
    save_dummy_method(
        "sequential_dummy",
        args.hard_budget,
        args.dataset_train_path,
        args.dataset_val_path,
    )

    save_dummy_method(
        "random_dummy",
        args.hard_budget,
        args.dataset_train_path,
        args.dataset_val_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models_folder",
        type=Path,
        default="models",
        help="Path to the models folder",
    )
    parser.add_argument(
        "--dataset_train_path",
        type=Path,
        default="data/cube/train_split_1.pt",
        help="Which dataset the model is trained on",
    )
    parser.add_argument(
        "--dataset_val_path",
        type=Path,
        default="data/cube/val_split_1.pt",
        help="Which dataset the model is validated on",
    )
    parser.add_argument(
        "--hard_budget",
        type=int,
        default=10,
        help="Hard budget for the method",
    )
    args = parser.parse_args()
    main(args)
