from pathlib import Path
from typing import Any
import torch
from torch import Tensor
import yaml

from common.custom_types import AFADataset, AFAMethod, FeatureMask

def get_afa_methods_with_fixed_keys(afa_method_name: str, fixed_key_mapping: dict[str, Any], results_path=Path("results"), models_path=Path("models")) -> list[Tensor]:
    """
    Return all evaluation results (as Tensors) for AFAMethods of a specific type that have been trained with a specific set of keys fixed to specific values.
    The remaining keys are allowed to take any value.
    """

    # Go through all folders in results_path / afa_method_name

    method_results_dir = results_path / afa_method_name
    if not method_results_dir.exists():
        raise ValueError(f"Results directory {method_results_dir} does not exist.")

    valid_instance_results: list[Tensor] = []
    for instance_results_path in method_results_dir.iterdir():
        # Open the corresponding params.yml file for model training (located in models_path)
        with open(models_path / afa_method_name / instance_results_path.name / "params.yml", "r") as file:
            train_params_dict: dict = yaml.safe_load(file)

        # Check if the keys match
        for key, value in fixed_key_mapping.items():
            if key not in train_params_dict or train_params_dict[key] != value:
                break
        else:
            # If we didn't break, all keys matched. Add the results tensor
            valid_instance_results.append(torch.load(instance_results_path / "results.pt"))

    return valid_instance_results
