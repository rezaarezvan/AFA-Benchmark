from types import SimpleNamespace
from typing import Any, Callable, Tuple

import torch
from jaxtyping import Integer
from torch import Tensor, nn

from afa_rl.custom_types import FeatureMask, FeatureSet
from common.custom_types import Features, MaskedFeatures


def remove_module_prefix(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def get_feature_set(
    masked_features: MaskedFeatures, feature_mask: FeatureMask
) -> FeatureSet:
    """
    Converts partially observed features and their indices to the state representation expected by the embedder.
    """
    batch_size, feature_size = masked_features.shape

    feature_set = torch.zeros(
        (batch_size, feature_size, 1 + feature_size),
        device=masked_features.device,
        dtype=torch.float,
    )

    # First column: feature values
    feature_set[:, :, 0] = masked_features

    # Generate one-hot indices
    one_hot_indices = torch.eye(
        feature_size, device=masked_features.device
    )  # shape: (feature_size, feature_size)

    # Expand one-hot vectors only for observed features
    mask_expanded = feature_mask.unsqueeze(-1).expand(
        -1, -1, feature_size
    )  # shape: (batch, features, feature_size)
    feature_set[:, :, 1:] = (
        one_hot_indices.unsqueeze(0).expand(batch_size, -1, -1) * mask_expanded
    )

    return feature_set


def get_image_feature_set(
    masked_image: MaskedFeatures,
    feature_mask: FeatureMask,
    image_shape: Tuple[int, int],
) -> FeatureSet:
    """
    Converts a partially observed image and the indices to a feature set.

    The output has shape (batch, HxW, 3), where each row in the third dimension is
    [value, row, col] for observed pixels in each batch. Unobserved pixels are
    included with value=0, row=col=0 and are padded to the end of the output.
    """
    batch_size, flat_dim = masked_image.shape
    h, w = image_shape

    # Reshape into (B, H, W)
    masked_image = masked_image.view(batch_size, h, w)
    feature_mask = feature_mask.view(batch_size, h, w)

    # Prepare a tensor to hold the result
    result = []

    for i in range(batch_size):
        # Get observed pixel indices (n_features, 2)
        obs_indices = torch.nonzero(
            feature_mask[i], as_tuple=False
        )  # shape (n_features, 2)
        values = masked_image[i, obs_indices[:, 0], obs_indices[:, 1]]

        # Adjust the row and column indices to 1-based indexing
        obs_indices = obs_indices + 1  # Make row and col 1-based

        # Stack [value, row, col] for observed pixels
        feature_set = torch.stack(
            [values, obs_indices[:, 0].float(), obs_indices[:, 1].float()], dim=1
        )

        # Create a tensor of size (H×W, 3) by padding unobserved pixels with zeros
        padded_set = torch.zeros((h * w, 3), device=masked_image.device)

        # For unobserved pixels, set value=0 and row=col=0
        padded_set[:, 0] = 0  # value = 0
        padded_set[:, 1] = 0  # row = 0
        padded_set[:, 2] = 0  # col = 0

        # Copy observed pixels into the result
        padded_set[: feature_set.size(0), :] = feature_set

        result.append(padded_set)

    return torch.stack(result)


def get_2D_identity(
    feature_mask: FeatureMask,
    image_shape: Tuple[int, int],
) -> Integer[Tensor, "*batch n_features 2"]:
    """
    Returns the coordinates for each observed feature (as given by the feature mask)
    but with (0, 0) for unobserved features.
    """
    batch_size, flat_dim = feature_mask.shape
    h, w = image_shape

    # Reshape into (B, H, W)
    feature_mask = feature_mask.view(batch_size, h, w)

    # Prepare a tensor to hold the result
    result = []

    for i in range(batch_size):
        # Get observed pixel indices (n_features, 2)
        obs_indices = torch.nonzero(
            feature_mask[i], as_tuple=False
        )  # shape (n_features, 2)

        # Adjust the row and column indices to 1-based indexing
        obs_indices = obs_indices + 1  # Make row and col 1-based

        # Stack [row, col] for observed pixels
        coords = torch.stack(
            [obs_indices[:, 0].float(), obs_indices[:, 1].float()], dim=1
        )

        # Create a tensor of size (H×W, 2) by padding unobserved pixels with zeros
        padded_set = torch.zeros((h * w, 2), device=feature_mask.device)

        # For unobserved pixels, set row=col=0
        padded_set[:, 0] = 0  # row = 0
        padded_set[:, 1] = 0  # col = 0

        # Copy observed pixels into the result
        padded_set[: coords.size(0), :] = coords

        result.append(padded_set)

    return torch.stack(result)


def get_1D_identity(
    feature_mask: FeatureMask,
) -> Integer[Tensor, "*batch n_features n_features"]:
    """
    Converts a feature mask to a onehot representation for each feature, but with all zeros for unobserved features.
    """
    batch_size, feature_size = feature_mask.shape

    feature_set = torch.zeros(
        (batch_size, feature_size, feature_size),
        device=feature_mask.device,
        dtype=torch.float,
    )

    # Generate one-hot indices
    one_hot_indices = torch.eye(
        feature_size, device=feature_mask.device
    )  # shape: (feature_size, feature_size)

    # Expand one-hot vectors only for observed features
    mask_expanded = feature_mask.unsqueeze(-1).expand(
        -1, -1, feature_size
    )  # shape: (batch, features, feature_size)
    feature_set[:, :, :] = (
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
    valid_action_values[~action_mask] = float("-inf")  # Mask out invalid actions
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


def mask_data(features: Features, p: float) -> Tuple[MaskedFeatures, FeatureMask]:
    """
    Given features, mask them with probability p.
    Returns the masked features and the mask.

    Args:
        batch: The features to mask.
        p: The probability of each feature being masked (0).
    """
    feature_mask = torch.rand(features.shape, device=features.device) > p
    masked_features = features * feature_mask.float()
    return masked_features, feature_mask
