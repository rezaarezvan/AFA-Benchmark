from pathlib import Path
from typing import Any
import torch

from common.utils import get_folders_with_matching_params

from pathlib import Path
from typing import Any

import torch


def get_eval_results_with_fixed_keys(fixed_params_mapping: dict[str, Any]={}, results_path=Path("results")) -> list[dict[str, Any]]:
    """
    Return all evaluation results (as dictionaries) that have specific params values.
    The remaining keys are allowed to take any value.

    Args:
        fixed_params_mapping (dict[str, Any]): A dictionary mapping parameter names to their fixed values. Applies to the `params.yml` file in the results folder.
        results_path (Path): The path to the results folder. Defaults to "results".
    """

    return [torch.load(folder / "results.pt") for folder in get_folders_with_matching_params(results_path, fixed_params_mapping)]

def get_classifier_paths_trained_on_data(classifier_type: str, train_dataset_path: Path, classifier_folder=Path("models/classifiers")) -> list[Path]:
    """
    Get Paths to all classifiers of a specific type trained on a specific dataset.
    """

    # Define the fixed parameters to match
    fixed_params_mapping = {"train_dataset_path": str(train_dataset_path)}

    # Get all matching folders
    matching_folders = get_folders_with_matching_params(classifier_folder / classifier_type, fixed_params_mapping)

    return matching_folders
