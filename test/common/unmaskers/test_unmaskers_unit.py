import pytest
import torch

from afabench.common.afa_unmaskers import DirectUnmasker, ImagePatchUnmasker
from afabench.common.config_classes import ImagePatchUnmaskerConfig
from afabench.common.custom_types import (
    AFASelection,
    FeatureMask,
    Features,
    MaskedFeatures,
    SelectionMask,
)


# Fixtures
@pytest.fixture
def fixture() -> tuple[
    Features,
    FeatureMask,
    MaskedFeatures,
    AFASelection,
    SelectionMask,
    torch.Size,
]:
    """Synthetic data for testing."""
    batch_size = 2
    feature_shape = torch.Size([3, 3])
    features = torch.randn(batch_size, *feature_shape).float()
    feature_mask = torch.zeros(batch_size, *feature_shape, dtype=torch.bool)
    masked_features = features * feature_mask
    afa_selection = torch.tensor(
        [
            [1],  # Select feature 1 for the first batch
            [3],  # Select feature 3 for the second batch
        ]
    )
    selection_mask = torch.ones_like(afa_selection, dtype=torch.bool)
    return (
        features,
        feature_mask,
        masked_features,
        afa_selection,
        selection_mask,
        feature_shape,
    )


# Test DirectUnmasker
def test_direct_unmasker(
    fixture: tuple[
        Features,
        FeatureMask,
        MaskedFeatures,
        AFASelection,
        SelectionMask,
        torch.Size,
    ],
) -> None:
    """Test DirectUnmasker unmasking behavior."""
    (
        features,
        feature_mask,
        masked_features,
        afa_selection,
        selection_mask,
        feature_shape,
    ) = fixture

    unmasker = DirectUnmasker()
    new_feature_mask = unmasker.unmask(
        masked_features=masked_features,
        feature_mask=feature_mask,
        features=features,
        afa_selection=afa_selection,
        selection_mask=selection_mask,
        feature_shape=feature_shape,
    )

    # Verify the new feature mask
    assert new_feature_mask.shape == feature_mask.shape
    assert new_feature_mask[0, 0, 0]  # Feature 1 in batch 0 is unmasked
    assert not new_feature_mask[0, 0, 1]  # Feature 2 in batch 0 remains masked
    assert not new_feature_mask[0, 0, 2]  # Feature 3 in batch 0 remains masked
    assert new_feature_mask[1, 0, 2]  # Feature 3 in batch 1 is unmasked


# Test ImagePatchUnmasker
@pytest.fixture
def image_patch_fixture() -> tuple[
    Features,
    FeatureMask,
    MaskedFeatures,
    AFASelection,
    SelectionMask,
    torch.Size,
    ImagePatchUnmaskerConfig,
]:
    """Synthetic data for ImagePatchUnmasker testing."""
    batch_size = 2
    image_side_length = 8
    patch_size = 4
    n_channels = 1

    feature_shape = torch.Size(
        [n_channels, image_side_length, image_side_length]
    )
    features = torch.randn(batch_size, *feature_shape).float()
    feature_mask = torch.zeros(batch_size, *feature_shape, dtype=torch.bool)
    masked_features = features * feature_mask
    afa_selection = torch.tensor(
        [
            [1],  # Select patch 1 for the first batch
            [3],  # Select patch 3 for the second batch
        ]
    )
    selection_mask = torch.ones_like(afa_selection, dtype=torch.bool)

    config = ImagePatchUnmaskerConfig(
        image_side_length=image_side_length,
        patch_size=patch_size,
        n_channels=n_channels,
    )

    return (
        features,
        feature_mask,
        masked_features,
        afa_selection,
        selection_mask,
        feature_shape,
        config,
    )


def test_image_patch_unmasker(
    image_patch_fixture: tuple[
        Features,
        FeatureMask,
        MaskedFeatures,
        AFASelection,
        SelectionMask,
        torch.Size,
        ImagePatchUnmaskerConfig,
    ],
) -> None:
    """Test ImagePatchUnmasker unmasking behavior."""
    (
        features,
        feature_mask,
        masked_features,
        afa_selection,
        selection_mask,
        feature_shape,
        config,
    ) = image_patch_fixture

    unmasker = ImagePatchUnmasker(config)
    new_feature_mask = unmasker.unmask(
        masked_features=masked_features,
        feature_mask=feature_mask,
        features=features,
        afa_selection=afa_selection,
        selection_mask=selection_mask,
        feature_shape=feature_shape,
    )

    # Verify the new feature mask
    assert new_feature_mask.shape == feature_mask.shape

    # Check that the correct patches are unmasked
    patch_size = config.patch_size
    assert new_feature_mask[0, :, :patch_size, :patch_size].all()  # Patch 1
    assert not new_feature_mask[
        0, :, patch_size:, :
    ].any()  # Other patches remain masked
    assert new_feature_mask[
        1, :, patch_size : 2 * patch_size, :patch_size
    ].all()  # Patch 3
