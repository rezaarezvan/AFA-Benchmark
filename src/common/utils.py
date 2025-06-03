def set_seed(seed: int):
    import os
    import random
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


from pathlib import Path
from types import SimpleNamespace
from typing import Any
from jaxtyping import Float, Bool
from torch import Tensor
import torch
import wandb
import yaml

from common.custom_types import AFADataset
from common.registry import get_afa_dataset_class


def get_class_probabilities(
    labels: Bool[Tensor, "*batch n_classes"],
) -> Float[Tensor, "n_classes"]:
    """
    Returns the class probabilities for a given set of labels.
    """
    class_counts = labels.float().sum(dim=0)
    class_probabilities = class_counts / class_counts.sum()
    return class_probabilities


def yaml_file_matches_mapping(yaml_file_path: Path, mapping: dict[str, Any]) -> bool:
    """
    Check if the keys in a YAML file match a given mapping (for the provided keys, other keys can have any value).
    """

    with open(yaml_file_path, "r") as file:
        dictionary: dict = yaml.safe_load(file)

    # Check if the keys match
    for key, value in mapping.items():
        if key not in dictionary or dictionary[key] != value:
            return False

    return True


def get_folders_with_matching_params(
    folder: Path, mapping: dict[str, Any]
) -> list[Path]:
    """
    Get all folders in a given folder that have a matching params.yml file.
    """

    matching_folders = [
        f
        for f in folder.iterdir()
        if yaml_file_matches_mapping(f / "params.yml", mapping)
    ]

    return matching_folders


def dict_to_namespace(d: Any) -> SimpleNamespace:
    """Convert a dict to a SimpleNamespace recursively."""
    if not isinstance(d, dict):
        return d

    # Create a namespace for this level
    ns = SimpleNamespace()

    # Convert each key-value pair
    for key, value in d.items():
        if isinstance(value, dict):
            # Recursively convert nested dictionaries
            setattr(ns, key, dict_to_namespace(value))
        elif isinstance(value, list):
            # Convert lists with potential nested dictionaries
            setattr(
                ns,
                key,
                [
                    dict_to_namespace(item) if isinstance(item, dict) else item
                    for item in value
                ],
            )
        else:
            # Set the attribute directly for primitive types
            setattr(ns, key, value)

    return ns


def load_dataset_artifact(
    artifact_name: str,
) -> tuple[AFADataset, AFADataset, AFADataset, dict[str, Any]]:
    """Load train, validation, and test datasets from a WandB artifact, together with its metadata."""
    dataset_artifact = wandb.use_artifact(artifact_name, type="dataset")
    dataset_artifact_dir = Path(dataset_artifact.download())
    # The dataset dir should contain the files train.pt, val.pt and test.pt
    artifact_filenames = [f.name for f in dataset_artifact_dir.iterdir()]
    assert {"train.pt", "val.pt", "test.pt"}.issubset(artifact_filenames), (
        f"Dataset artifact must contain train.pt, val.pt and test.pt files. Instead found: {artifact_filenames}"
    )

    dataset_class = get_afa_dataset_class(dataset_artifact.metadata["dataset_type"])
    train_dataset: AFADataset = dataset_class.load(dataset_artifact_dir / "train.pt")
    val_dataset: AFADataset = dataset_class.load(dataset_artifact_dir / "val.pt")
    test_dataset: AFADataset = dataset_class.load(dataset_artifact_dir / "test.pt")

    return train_dataset, val_dataset, test_dataset, dataset_artifact.metadata
