"""
Generates .pt files for the two dummy methods SequentialDummyAFAMethod and RandomDummyAFAMethod
"""


import argparse
import torch
from tqdm import tqdm
import os

from afa_rl.afa_methods import RandomDummyAFAMethod, SequentialDummyAFAMethod


def main(args: argparse.Namespace):
    # Create args.models_folder if necessary
    os.makedirs(args.models_folder, exist_ok=True)
    # Create subfolders for the two methods if necessary
    os.makedirs(os.path.join(args.models_folder, "sequential_dummy"), exist_ok=True)
    os.makedirs(os.path.join(args.models_folder, "random_dummy"), exist_ok=True)
    for i in tqdm(range(1, 6)):
        method = SequentialDummyAFAMethod(
            device=torch.device("cpu"),
            n_classes=args.n_classes,
        )
        method.save(
            os.path.join(
                args.models_folder,
                "sequential_dummy",
                f"sequential_dummy-{args.dataset_type}_train_split_{i}.pt",
            )
        )

        method = RandomDummyAFAMethod(
            device=torch.device("cpu"),
            n_classes=args.n_classes,
        )
        method.save(
            os.path.join(
                args.models_folder,
                "random_dummy",
                f"random_dummy-{args.dataset_type}_train_split_{i}.pt",
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models_folder",
        type=str,
        required=True,
        help="Path to the models folder",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        required=True,
        help="Name of the dataset",
    )
    parser.add_argument(
        "--n_classes",
        type=int,
        required=True,
        help="Number of classes in the dataset",
    )
    args = parser.parse_args()
    main(args)
