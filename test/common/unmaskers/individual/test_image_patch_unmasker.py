import pytest
import torch

from afabench.common.config_classes import ImagePatchUnmaskerConfig
from afabench.common.custom_types import (
    AFASelection,
    FeatureMask,
    Features,
    MaskedFeatures,
    SelectionMask,
)
from afabench.common.unmaskers import ImagePatchUnmasker


@pytest.fixture
def basic_image_fixture() -> tuple[
    Features,
    FeatureMask,
    MaskedFeatures,
    AFASelection,
    SelectionMask,
    torch.Size,
    ImagePatchUnmaskerConfig,
]:
    """Provide basic test data for ImagePatchUnmasker."""
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


def test_image_patch_unmasker_basic_functionality(
    basic_image_fixture: tuple[
        Features,
        FeatureMask,
        MaskedFeatures,
        AFASelection,
        SelectionMask,
        torch.Size,
        ImagePatchUnmaskerConfig,
    ],
) -> None:
    """Test basic patch unmasking behavior."""
    (
        features,
        feature_mask,
        masked_features,
        afa_selection,
        selection_mask,
        feature_shape,
        config,
    ) = basic_image_fixture

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


def test_image_patch_unmasker_arbitrary_batch_shape() -> None:
    """Test ImagePatchUnmasker with arbitrary batch shapes."""
    batch_shape = torch.Size([2, 3])  # Multi-dimensional batch
    image_side_length = 8
    patch_size = 4
    n_channels = 2

    feature_shape = torch.Size(
        [n_channels, image_side_length, image_side_length]
    )
    features = torch.randn(*batch_shape, *feature_shape)
    feature_mask = torch.zeros(*batch_shape, *feature_shape, dtype=torch.bool)
    masked_features = features * feature_mask

    # Select patches for each batch element (4 patches total: 2x2 grid)
    afa_selection = torch.randint(1, 5, (*batch_shape, 1))  # Patches 1-4
    selection_mask = torch.ones_like(afa_selection, dtype=torch.bool)

    config = ImagePatchUnmaskerConfig(
        image_side_length=image_side_length,
        patch_size=patch_size,
        n_channels=n_channels,
    )

    unmasker = ImagePatchUnmasker(config)
    new_feature_mask = unmasker.unmask(
        masked_features=masked_features,
        feature_mask=feature_mask,
        features=features,
        afa_selection=afa_selection,
        selection_mask=selection_mask,
        feature_shape=feature_shape,
    )

    # Verify output shape
    assert new_feature_mask.shape == (*batch_shape, *feature_shape)

    # Verify that exactly one patch is unmasked per batch element
    patch_pixels = patch_size * patch_size * n_channels
    new_feature_mask_flat = new_feature_mask.view(-1, *feature_shape)
    for i in range(new_feature_mask_flat.shape[0]):
        assert new_feature_mask_flat[i].sum() == patch_pixels


def test_image_patch_unmasker_zero_selection() -> None:
    """Test ImagePatchUnmasker with zero selection (no unmasking)."""
    batch_size = 3
    image_side_length = 4
    patch_size = 2
    n_channels = 1

    feature_shape = torch.Size(
        [n_channels, image_side_length, image_side_length]
    )
    features = torch.randn(batch_size, *feature_shape)
    feature_mask = torch.zeros(batch_size, *feature_shape, dtype=torch.bool)
    masked_features = features * feature_mask

    # Select patch 0 (which means no selection)
    afa_selection = torch.zeros(batch_size, 1, dtype=torch.long)
    selection_mask = torch.ones_like(afa_selection, dtype=torch.bool)

    config = ImagePatchUnmaskerConfig(
        image_side_length=image_side_length,
        patch_size=patch_size,
        n_channels=n_channels,
    )

    unmasker = ImagePatchUnmasker(config)
    new_feature_mask = unmasker.unmask(
        masked_features=masked_features,
        feature_mask=feature_mask,
        features=features,
        afa_selection=afa_selection,
        selection_mask=selection_mask,
        feature_shape=feature_shape,
    )

    # Verify no features are unmasked when selection is 0
    assert torch.equal(new_feature_mask, feature_mask)
    assert new_feature_mask.sum() == 0


def test_image_patch_unmasker_preserves_existing_mask() -> None:
    """Test that existing mask is preserved and extended."""
    batch_size = 2
    image_side_length = 6
    patch_size = 3
    n_channels = 1

    feature_shape = torch.Size(
        [n_channels, image_side_length, image_side_length]
    )
    features = torch.randn(batch_size, *feature_shape)

    # Start with some features already unmasked
    feature_mask = torch.zeros(batch_size, *feature_shape, dtype=torch.bool)
    feature_mask[0, :, 0:2, 0:2] = True  # Pre-existing unmasked region
    feature_mask[1, :, 3:5, 3:5] = True  # Pre-existing unmasked region

    masked_features = features * feature_mask

    # Select patches to unmask
    afa_selection = torch.tensor([[2], [1]])  # Different patches
    selection_mask = torch.ones_like(afa_selection, dtype=torch.bool)

    config = ImagePatchUnmaskerConfig(
        image_side_length=image_side_length,
        patch_size=patch_size,
        n_channels=n_channels,
    )

    unmasker = ImagePatchUnmasker(config)
    new_feature_mask = unmasker.unmask(
        masked_features=masked_features,
        feature_mask=feature_mask,
        features=features,
        afa_selection=afa_selection,
        selection_mask=selection_mask,
        feature_shape=feature_shape,
    )

    # Verify that original unmasked regions are preserved
    assert new_feature_mask[
        0, :, 0:2, 0:2
    ].all()  # Original region still unmasked
    assert new_feature_mask[
        1, :, 3:5, 3:5
    ].all()  # Original region still unmasked

    # Verify that new patches are also unmasked
    assert new_feature_mask[
        0, :, 0:3, 3:6
    ].all()  # Patch 2 unmasked in batch 0
    assert new_feature_mask[
        1, :, 0:3, 0:3
    ].all()  # Patch 1 unmasked in batch 1


def test_image_patch_unmasker_get_n_selections() -> None:
    """Test get_n_selections method."""
    config = ImagePatchUnmaskerConfig(
        image_side_length=8,
        patch_size=4,
        n_channels=3,
    )
    unmasker = ImagePatchUnmasker(config)

    # Should have 2x2 = 4 patches
    feature_shape = torch.Size([3, 8, 8])
    assert unmasker.get_n_selections(feature_shape) == 4

    # Test with different configuration
    config2 = ImagePatchUnmaskerConfig(
        image_side_length=12,
        patch_size=3,
        n_channels=1,
    )
    unmasker2 = ImagePatchUnmasker(config2)

    # Should have 4x4 = 16 patches
    feature_shape2 = torch.Size([1, 12, 12])
    assert unmasker2.get_n_selections(feature_shape2) == 16


def test_image_patch_unmasker_different_patch_sizes() -> None:
    """Test with different patch sizes."""
    batch_size = 2
    image_side_length = 12
    n_channels = 1

    patch_sizes = [2, 3, 4, 6]

    for patch_size in patch_sizes:
        feature_shape = torch.Size(
            [n_channels, image_side_length, image_side_length]
        )
        features = torch.randn(batch_size, *feature_shape)
        feature_mask = torch.zeros(
            batch_size, *feature_shape, dtype=torch.bool
        )
        masked_features = features * feature_mask

        # Select first patch
        afa_selection = torch.ones(batch_size, 1, dtype=torch.long)
        selection_mask = torch.ones_like(afa_selection, dtype=torch.bool)

        config = ImagePatchUnmaskerConfig(
            image_side_length=image_side_length,
            patch_size=patch_size,
            n_channels=n_channels,
        )

        unmasker = ImagePatchUnmasker(config)
        new_feature_mask = unmasker.unmask(
            masked_features=masked_features,
            feature_mask=feature_mask,
            features=features,
            afa_selection=afa_selection,
            selection_mask=selection_mask,
            feature_shape=feature_shape,
        )

        # Verify correct patch size is unmasked
        expected_patch_pixels = patch_size * patch_size * n_channels
        assert new_feature_mask.sum() == expected_patch_pixels * batch_size


def test_image_patch_unmasker_multichannel() -> None:
    """Test with multi-channel images."""
    batch_size = 3
    image_side_length = 8
    patch_size = 4
    n_channels = 3  # RGB

    feature_shape = torch.Size(
        [n_channels, image_side_length, image_side_length]
    )
    features = torch.randn(batch_size, *feature_shape)
    feature_mask = torch.zeros(batch_size, *feature_shape, dtype=torch.bool)
    masked_features = features * feature_mask

    # Select different patches
    afa_selection = torch.tensor([[1], [2], [4]])
    selection_mask = torch.ones_like(afa_selection, dtype=torch.bool)

    config = ImagePatchUnmaskerConfig(
        image_side_length=image_side_length,
        patch_size=patch_size,
        n_channels=n_channels,
    )

    unmasker = ImagePatchUnmasker(config)
    new_feature_mask = unmasker.unmask(
        masked_features=masked_features,
        feature_mask=feature_mask,
        features=features,
        afa_selection=afa_selection,
        selection_mask=selection_mask,
        feature_shape=feature_shape,
    )

    # Verify that all channels are unmasked for selected patches
    patch_pixels = patch_size * patch_size
    for i in range(batch_size):
        # Each batch element should have one patch unmasked across all channels
        assert new_feature_mask[i].sum() == patch_pixels * n_channels


def test_image_patch_unmasker_all_patches() -> None:
    """Test unmasking all possible patches."""
    batch_size = 4  # Same as number of patches
    image_side_length = 4
    patch_size = 2
    n_channels = 1

    feature_shape = torch.Size(
        [n_channels, image_side_length, image_side_length]
    )
    features = torch.randn(batch_size, *feature_shape)
    feature_mask = torch.zeros(batch_size, *feature_shape, dtype=torch.bool)
    masked_features = features * feature_mask

    # Select all patches (1, 2, 3, 4)
    afa_selection = torch.tensor([[1], [2], [3], [4]])
    selection_mask = torch.ones_like(afa_selection, dtype=torch.bool)

    config = ImagePatchUnmaskerConfig(
        image_side_length=image_side_length,
        patch_size=patch_size,
        n_channels=n_channels,
    )

    unmasker = ImagePatchUnmasker(config)
    new_feature_mask = unmasker.unmask(
        masked_features=masked_features,
        feature_mask=feature_mask,
        features=features,
        afa_selection=afa_selection,
        selection_mask=selection_mask,
        feature_shape=feature_shape,
    )

    # Verify that different patches are unmasked for each batch element
    patch_pixels = patch_size * patch_size
    for i in range(batch_size):
        assert new_feature_mask[i].sum() == patch_pixels

    # Verify that when all patches are unmasked across batches, the entire image is covered
    combined_mask = new_feature_mask.sum(dim=0) > 0
    assert combined_mask.all()


def test_image_patch_unmasker_configuration_validation() -> None:
    """Test configuration validation."""
    # Valid configuration
    config = ImagePatchUnmaskerConfig(
        image_side_length=8,
        patch_size=4,
        n_channels=1,
    )
    ImagePatchUnmasker(config)  # Should not raise

    # Invalid configuration - image not divisible by patch size
    invalid_config = ImagePatchUnmaskerConfig(
        image_side_length=9,  # Not divisible by 4
        patch_size=4,
        n_channels=1,
    )
    with pytest.raises(AssertionError, match="divisible by patch size"):
        ImagePatchUnmasker(invalid_config)


def test_image_patch_unmasker_large_image() -> None:
    """Test with larger images and smaller patches."""
    batch_size = 5
    image_side_length = 16
    patch_size = 4
    n_channels = 2

    feature_shape = torch.Size(
        [n_channels, image_side_length, image_side_length]
    )
    features = torch.randn(batch_size, *feature_shape)
    feature_mask = torch.zeros(batch_size, *feature_shape, dtype=torch.bool)
    masked_features = features * feature_mask

    # Should have 4x4 = 16 possible patches
    max_patch_selection = (image_side_length // patch_size) ** 2

    # Select random patches
    afa_selection = torch.randint(1, max_patch_selection + 1, (batch_size, 1))
    selection_mask = torch.ones_like(afa_selection, dtype=torch.bool)

    config = ImagePatchUnmaskerConfig(
        image_side_length=image_side_length,
        patch_size=patch_size,
        n_channels=n_channels,
    )

    unmasker = ImagePatchUnmasker(config)
    new_feature_mask = unmasker.unmask(
        masked_features=masked_features,
        feature_mask=feature_mask,
        features=features,
        afa_selection=afa_selection,
        selection_mask=selection_mask,
        feature_shape=feature_shape,
    )

    # Verify that exactly one patch per batch element is unmasked
    patch_pixels = patch_size * patch_size * n_channels
    for i in range(batch_size):
        assert new_feature_mask[i].sum() == patch_pixels


def test_image_patch_unmasker_set_seed() -> None:
    """Test that set_seed method exists and is callable."""
    config = ImagePatchUnmaskerConfig(
        image_side_length=8,
        patch_size=4,
        n_channels=1,
    )
    unmasker = ImagePatchUnmasker(config)

    # Should not raise any exceptions
    unmasker.set_seed(42)
    unmasker.set_seed(None)


def test_image_patch_unmasker_multidimensional_batch() -> None:
    """Test with complex multi-dimensional batch shapes."""
    batch_shape = torch.Size([2, 2, 3])
    image_side_length = 6
    patch_size = 3
    n_channels = 1

    feature_shape = torch.Size(
        [n_channels, image_side_length, image_side_length]
    )
    features = torch.randn(*batch_shape, *feature_shape)
    feature_mask = torch.zeros(*batch_shape, *feature_shape, dtype=torch.bool)
    masked_features = features * feature_mask

    # Select patches (should have 2x2 = 4 patches available)
    afa_selection = torch.randint(1, 5, (*batch_shape, 1))
    selection_mask = torch.ones_like(afa_selection, dtype=torch.bool)

    config = ImagePatchUnmaskerConfig(
        image_side_length=image_side_length,
        patch_size=patch_size,
        n_channels=n_channels,
    )

    unmasker = ImagePatchUnmasker(config)
    new_feature_mask = unmasker.unmask(
        masked_features=masked_features,
        feature_mask=feature_mask,
        features=features,
        afa_selection=afa_selection,
        selection_mask=selection_mask,
        feature_shape=feature_shape,
    )

    assert new_feature_mask.shape == (*batch_shape, *feature_shape)

    # Check that exactly one patch per batch element is unmasked
    patch_pixels = patch_size * patch_size * n_channels
    total_batch_elements = torch.prod(torch.tensor(batch_shape))
    assert new_feature_mask.sum() == patch_pixels * total_batch_elements
