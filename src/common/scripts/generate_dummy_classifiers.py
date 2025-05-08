"""
Generates .pt files for the dummy classifier DummyAFAClassifier
"""


import argparse
from pathlib import Path
import torch

import yaml

from afa_rl.afa_methods import RandomDummyAFAMethod, SequentialDummyAFAMethod
from common.registry import AFA_CLASSIFIER_REGISTRY, AFA_DATASET_REGISTRY, AFA_METHOD_REGISTRY
from coolname import generate_slug


def save_dummy_classifier(classifier_name: str, dataset_train_path: Path, dataset_val_path: Path, seed: int, classifier_folder: Path):
    # Find classifier class using registry
    method_cls = AFA_CLASSIFIER_REGISTRY[classifier_name]

    # Find dataset type from the dataset path
    dataset_type = dataset_train_path.parent.name
    # Find dataset class using registry
    dataset_cls = AFA_DATASET_REGISTRY[dataset_type]

    # Generate a unique datetime string and create folders
    timestr = generate_slug(2)
    (classifier_folder / classifier_name / timestr).mkdir(
        parents=True, exist_ok=True
    )

    method = method_cls(
        n_classes=dataset_cls.n_classes,
    )

    method.save(
        classifier_folder / classifier_name / timestr / "model.pt"
    )
    # Also save a yaml file with params
    with open(
        (classifier_folder / classifier_name / timestr / "params.yml"), "w"
    ) as f:
        yaml.dump({
            "seed": seed,
            "dataset_type": dataset_type,
            "train_dataset_path": str(dataset_train_path),
            "val_dataset_path": str(dataset_val_path),
        }, f, default_flow_style=False)



def main(args: argparse.Namespace):
    # The dataset type is the name of the folder containing the dataset
    args.dataset_type = args.dataset_folder.name

    print("Generating dummy classifiers...")

    # Create a classifier for each (train_path, val_path) and seed combination
    for split in range(1, args.n_splits+1):
        train_path = args.dataset_folder / f"train_split_{split}.pt"
        val_path = args.dataset_folder / f"val_split_{split}.pt"

        # The data should exist
        assert train_path.exists() and val_path.exists()

        for seed in range(1, args.n_seeds+1):
            save_dummy_classifier(
                "random_dummy",
                train_path,
                val_path,
                seed,
                args.classifier_folder,
            )

            save_dummy_classifier(
                "uniform_dummy",
                train_path,
                val_path,
                seed,
                args.classifier_folder,
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classifier_folder",
        type=Path,
        default="models/classifiers",
        help="Path to the classifier folder",
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
        default=3,
        help="Number of splits for the dataset",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=1,
        help="How many different seeds to use",
    )
    args = parser.parse_args()
    main(args)
