from typing import Any, Callable

import torch

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
