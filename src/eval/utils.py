from pathlib import Path
from typing import Any
import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
import yaml

from common.custom_types import AFAClassifier, AFADataset, AFAMethod, FeatureMask
from common.registry import AFA_CLASSIFIER_REGISTRY

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

def get_eval_results_with_fixed_keys(fixed_params_mapping: dict[str, Any]={}, results_path=Path("results")) -> list[dict[str, Any]]:
    """
    Return all evaluation results (as dictionaries) that have specific params values.
    The remaining keys are allowed to take any value.

    Args:
        fixed_params_mapping (dict[str, Any]): A dictionary mapping parameter names to their fixed values. Applies to the `params.yml` file in the results folder.
        results_path (Path): The path to the results folder. Defaults to "results".
    """

    # Go through all folders in results_path

    valid_instance_results: list[dict[str, float]] = []
    for instance_results_path in results_path.iterdir():
        # Check if the results params.yml file matches the fixed_params_mapping
        if not yaml_file_matches_mapping(
            instance_results_path / "params.yml", fixed_params_mapping
        ):
            continue

        # Mapping match, save results
        valid_instance_results.append(
            torch.load(instance_results_path / "results.pt")
        )

    return valid_instance_results

def get_classifier_paths_trained_on_data(classifier_type: str, train_dataset_path: Path, classifier_folder = Path("models/classifiers")) -> list[Path]:
    """
    Get Paths to all classifiers of a specific type trained on a specific dataset.
    """

    valid_classifier_paths: list[Path] = []

    # Loop through each trained instance
    for trained_instance_path in (classifier_folder / classifier_type).iterdir():
        # The params.yml file should contain which dataset was used
        params_path = trained_instance_path / "params.yml"
        with open(params_path, "r") as file:
            params_dict: dict = yaml.safe_load(file)

        # Return classifier if it was trained on the same dataset
        if params_dict["train_dataset_path"] == str(train_dataset_path):
            valid_classifier_paths.append(trained_instance_path)
    return valid_classifier_paths
