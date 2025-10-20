"""Different types of AFAUncoverFn."""

from torch.nn import functional as F

from common.custom_types import (
    AFASelection,
    AFAUncoverFn,
    FeatureMask,
    Features,
    MaskedFeatures,
)


# Implements AFAUncoverFn
def one_based_index_uncover_fn(
    masked_features: MaskedFeatures,  # noqa: ARG001
    feature_mask: FeatureMask,
    features: Features,
    afa_selection: AFASelection,
) -> tuple[MaskedFeatures, FeatureMask]:
    """
    Uncover the features assuming `afa_selection` are 1-based indices of the features to uncover.

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


def get_image_patch_uncover_fn(
    image_side_length: int, n_channels: int, patch_size: int
) -> AFAUncoverFn:
    """
    Return an AFAUncoverFn that uncovers features in an image patch.

    Args:
        image_side_length (int): The side length of the image.
        n_channels (int): The number of channels in the image.
        patch_size (int): The size of the patch.

    Returns:
        AFAUncoverFn: An AFAUncoverFn that uncovers features in an image patch.
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
        afa_selection_1d = F.one_hot(
            afa_selection - 1, num_classes=afa_selection_size
        )

        # Convert to low-dimensional image mask
        afa_selection_low_dim_image = afa_selection_1d.reshape(
            low_dim_image_side_length,
            low_dim_image_side_length,
        )

        # Add batch dimension and channel dimension, expand channels
        afa_selection_low_dim_image = (
            afa_selection_low_dim_image.unsqueeze(0)
            .unsqueeze(0)
            .expand(-1, n_channels, -1, -1)
        )

        # Upscale image mask, converting between float and bool
        afa_selection_image = F.interpolate(
            afa_selection_low_dim_image.float(),
            scale_factor=patch_size,
            mode="nearest-exact",
        ).bool()

        print(f"{afa_selection_image.shape}")
        print(f"{feature_mask.shape}")

        # Convert image mask to feature mask and add to previous feature mask
        new_feature_mask = feature_mask + afa_selection_image.flatten(
            start_dim=1
        )

        # Apply new feature mask on features
        new_masked_features = features * new_feature_mask

        return new_masked_features, new_feature_mask

    return f
