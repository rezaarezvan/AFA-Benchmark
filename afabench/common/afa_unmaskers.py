import torch
from torch.nn import functional as F

from afabench.common.custom_types import (
    AFASelection,
    AFAUnmaskFn,
    FeatureMask,
    Features,
    MaskedFeatures,
)


# Implements AFAUnmaskFn
def one_based_index_unmask_fn(
    masked_features: MaskedFeatures,  # noqa: ARG001
    feature_mask: FeatureMask,
    features: Features,
    afa_selection: AFASelection,
) -> tuple[MaskedFeatures, FeatureMask]:
    """
    Unmasks the features assuming `afa_selection` are 1-based indices of the features to uncover.

    afa_selection == 0 is ignored.
    """
    relevant_indices = afa_selection != 0
    feature_indices = afa_selection[relevant_indices] - 1

    new_feature_mask = feature_mask.clone()
    new_feature_mask[relevant_indices.nonzero().flatten(), feature_indices] = (
        True
    )
    new_masked_features = features.clone()
    new_masked_features.masked_fill_(~new_feature_mask, 0)

    return new_masked_features, new_feature_mask


def get_image_patch_unmask_fn(
    image_side_length: int, n_channels: int, patch_size: int
) -> AFAUnmaskFn:
    """
    Return an AFAUnmaskFn that unmasks features in an image patch.

    Args:
        image_side_length (int): The side length of the image.
        n_channels (int): The number of channels in the image.
        patch_size (int): The size of the patch.

    Returns:
        AFAUnmaskFn: An AFAUnmaskFn that unmasks features in an image patch.
    """
    assert image_side_length % patch_size == 0, (
        "Image side length must be divisible by patch size"
    )
    low_dim_image_side_length = image_side_length // patch_size
    afa_selection_size = low_dim_image_side_length**2

    # Implements AFAUncoverFn
    def f(
        masked_features: MaskedFeatures,  # noqa: ARG001
        feature_mask: FeatureMask,
        features: Features,
        afa_selection: AFASelection,
    ) -> tuple[MaskedFeatures, FeatureMask]:
        assert afa_selection.ndim == 1, (
            f"Expected 1D array, got {afa_selection.ndim}D array"
        )

        # Convert afa selection into 1D mask
        sel = afa_selection.to(torch.long)
        stop = sel == 0
        # avoid -1 for one hot
        sel = torch.where(stop, torch.ones_like(sel), sel)
        afa_selection_1d = F.one_hot(sel - 1, num_classes=afa_selection_size)
        afa_selection_1d[stop] = 0

        # Convert to low-dimensional image mask
        afa_selection_low_dim_image = afa_selection_1d.view(
            -1,
            low_dim_image_side_length,
            low_dim_image_side_length,
        )

        # Add batch dimension and channel dimension, expand channels
        afa_selection_low_dim_image = afa_selection_low_dim_image.unsqueeze(
            1
        ).expand(-1, n_channels, -1, -1)

        # Upscale image mask, converting between float and bool
        afa_selection_image = F.interpolate(
            afa_selection_low_dim_image.float(),
            scale_factor=patch_size,
            mode="nearest-exact",
        ).bool()

        # print(f"{afa_selection_image.shape}")
        # print(f"{feature_mask.shape}")

        # Convert image mask to feature mask and add to previous feature mask
        # new_feature_mask = feature_mask + afa_selection_image.flatten(
        #     start_dim=1
        # )
        new_feature_mask = feature_mask | afa_selection_image

        # Apply new feature mask on features
        new_masked_features = features * new_feature_mask

        return new_masked_features, new_feature_mask

    return f
