from types import SimpleNamespace
from typing import Any, Callable

import torch
from torch import nn

from afa_rl.custom_types import Feature, FeatureMask, FeatureSet


def get_feature_set(feature_values: Feature, feature_mask: FeatureMask) -> FeatureSet:
    """
    Converts partially observed features and their indices to the state representation expected by the embedder.
    """
    batch_size, feature_size = feature_values.shape

    feature_set = torch.zeros(
        (batch_size, feature_size, 1 + feature_size),
        device=feature_values.device,
        dtype=torch.float,
    )

    # First column: feature values
    feature_set[:, :, 0] = feature_values

    # Generate one-hot indices
    one_hot_indices = torch.eye(
        feature_size, device=feature_values.device
    )  # shape: (feature_size, feature_size)

    # Expand one-hot vectors only for observed features
    mask_expanded = feature_mask.unsqueeze(-1).expand(
        -1, -1, feature_size
    )  # shape: (batch, features, feature_size)
    feature_set[:, :, 1:] = (
        one_hot_indices.unsqueeze(0).expand(batch_size, -1, -1) * mask_expanded
    )

    return feature_set


def FloatWrapFn(f: Callable[..., Any]):
    """
    Wraps a function to convert all arguments to float before calling it.
    """

    def wrapper(*args):
        return f(*[arg.float() for arg in args])

    return wrapper


def dict_to_namespace(d):
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


def resample_invalid_actions(actions, action_mask, action_values):
    resampled_actions = actions.clone()

    # Find invalid actions
    invalid_mask = ~action_mask[torch.arange(actions.shape[0]), actions]

    # Select the highest-value valid action for each invalid case
    valid_action_values = action_values.clone()
    valid_action_values[~action_mask] = float('-inf')  # Mask out invalid actions
    best_valid_actions = valid_action_values.argmax(dim=-1)

    resampled_actions[invalid_mask] = best_valid_actions[invalid_mask]

    return resampled_actions



def get_sequential_module_norm(module: nn.Sequential):
    """
    Calculates the average norm of all the linear layers in a sequential module.
    """
    weight_norms = [
        layer.weight.norm() for layer in module if isinstance(layer, nn.Linear)
    ]
    return torch.mean(torch.stack(weight_norms)).item()
