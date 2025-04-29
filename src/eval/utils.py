from pathlib import Path
from typing import Any
import torch
from torch import Tensor
import yaml

from common.custom_types import AFADataset, AFAMethod, FeatureMask

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

def get_afa_methods_with_fixed_keys(afa_method_name: str, fixed_train_params_mapping: dict[str, Any]={}, fixed_eval_params_mapping: dict[str, Any]={}, results_path=Path("results"), models_path=Path("models")) -> list[Tensor]:
    """
    Return all evaluation results (as Tensors) for AFAMethods of a specific type that have specific params values (both training and evaluation).
    The remaining keys are allowed to take any value.

    Args:
        afa_method_name (str): The name of the AFA method to filter by. Must be in AFA_METHOD_REGISTRY.
        fixed_train_params_mapping (dict[str, Any]): A dictionary mapping parameter names to their fixed values for training. Applies to the `params.yml` file in the models folder.
        fixed_eval_params_mapping (dict[str, Any]): A dictionary mapping parameter names to their fixed values for evaluation. Applies to the `params.yml` file in the results folder.
        results_path (Path): The path to the results folder. Defaults to "results".
        models_path (Path): The path to the models folder. Defaults to "models".
    """

    # Go through all folders in results_path / afa_method_name

    method_results_dir = results_path / afa_method_name
    if not method_results_dir.exists():
        raise ValueError(f"Results directory {method_results_dir} does not exist.")

    valid_instance_results: list[Tensor] = []
    for instance_results_path in method_results_dir.iterdir():
        # Check if the results params.yml file matches the fixed_eval_params_mapping
        if not yaml_file_matches_mapping(
            instance_results_path / "params.yml", fixed_eval_params_mapping
        ):
            continue


        # Check if the model params.yml file matches the fixed_train_params_mapping
        if not yaml_file_matches_mapping(
            models_path / afa_method_name / instance_results_path.name / "params.yml",
            fixed_train_params_mapping,
        ):
            continue

        # Both mapping match, save results
        valid_instance_results.append(
            torch.load(instance_results_path / "results.pt")
        )

    return valid_instance_results
