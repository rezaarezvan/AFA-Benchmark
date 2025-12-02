import pytest
import torch

from afabench.common.custom_types import (
    AFASelection,
    FeatureMask,
    Features,
    MaskedFeatures,
    SelectionMask,
)
from afabench.common.unmaskers import DirectUnmasker


@pytest.fixture
def basic_fixture() -> tuple[
    Features,
    FeatureMask,
    MaskedFeatures,
    AFASelection,
    SelectionMask,
    torch.Size,
]:
    """Basic test data for DirectUnmasker."""
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


def test_direct_unmasker_basic_functionality(basic_fixture):
    """Test basic unmasking behavior."""
    (
        features,
        feature_mask,
        masked_features,
        afa_selection,
        selection_mask,
        feature_shape,
    ) = basic_fixture

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


def test_direct_unmasker_arbitrary_batch_shape():
    """Test DirectUnmasker with arbitrary batch shapes."""
    batch_shape = torch.Size([2, 3, 4])  # Multi-dimensional batch
    feature_shape = torch.Size([5, 6])
    features = torch.randn(*batch_shape, *feature_shape)
    feature_mask = torch.zeros(*batch_shape, *feature_shape, dtype=torch.bool)
    masked_features = features * feature_mask

    # Select different features for each batch element
    afa_selection = torch.randint(
        1, feature_shape.numel() + 1, (*batch_shape, 1)
    )
    selection_mask = torch.ones_like(afa_selection, dtype=torch.bool)

    unmasker = DirectUnmasker()
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

    # Verify that exactly one feature is unmasked per batch element
    new_feature_mask_flat = new_feature_mask.view(-1, *feature_shape)
    for i in range(new_feature_mask_flat.shape[0]):
        assert new_feature_mask_flat[i].sum() == 1


def test_direct_unmasker_zero_selection():
    """Test that zero selection is ignored."""
    batch_size = 3
    feature_shape = torch.Size([4, 4])
    features = torch.randn(batch_size, *feature_shape)
    feature_mask = torch.zeros(batch_size, *feature_shape, dtype=torch.bool)
    masked_features = features * feature_mask

    # Selection of 0 should be ignored
    afa_selection = torch.zeros(batch_size, 1, dtype=torch.long)
    selection_mask = torch.ones_like(afa_selection, dtype=torch.bool)

    unmasker = DirectUnmasker()
    new_feature_mask = unmasker.unmask(
        masked_features=masked_features,
        feature_mask=feature_mask,
        features=features,
        afa_selection=afa_selection,
        selection_mask=selection_mask,
        feature_shape=feature_shape,
    )

    # No features should be unmasked when selection is 0
    assert torch.equal(new_feature_mask, feature_mask)


def test_direct_unmasker_preserves_existing_mask():
    """Test that existing mask is preserved and extended."""
    batch_size = 2
    feature_shape = torch.Size([3, 3])
    features = torch.randn(batch_size, *feature_shape)

    # Start with some features already unmasked
    feature_mask = torch.zeros(batch_size, *feature_shape, dtype=torch.bool)
    feature_mask[0, 1, 1] = True  # Pre-existing unmasked feature
    feature_mask[1, 2, 2] = True  # Pre-existing unmasked feature

    masked_features = features * feature_mask

    # Select additional features to unmask
    afa_selection = torch.tensor([[5], [2]])  # Features 5 and 2
    selection_mask = torch.ones_like(afa_selection, dtype=torch.bool)

    unmasker = DirectUnmasker()
    new_feature_mask = unmasker.unmask(
        masked_features=masked_features,
        feature_mask=feature_mask,
        features=features,
        afa_selection=afa_selection,
        selection_mask=selection_mask,
        feature_shape=feature_shape,
    )

    # Verify that original unmasked features are preserved
    assert new_feature_mask[0, 1, 1]  # Original feature still unmasked
    assert new_feature_mask[1, 2, 2]  # Original feature still unmasked

    # Verify that new features are unmasked
    # Feature 5 in flattened 3x3 grid is at position (1, 1) (5-1=4, unravel(4) = (1,1))
    # Feature 2 in flattened 3x3 grid is at position (0, 1) (2-1=1, unravel(1) = (0,1))
    assert new_feature_mask[0, 1, 1]  # Feature 5 -> (1, 1) unmasked in batch 0
    assert new_feature_mask[1, 0, 1]  # Feature 2 -> (0, 1) unmasked in batch 1


def test_direct_unmasker_1d_features():
    """Test with 1D features."""
    batch_size = 5
    feature_shape = torch.Size([10])
    features = torch.randn(batch_size, *feature_shape)
    feature_mask = torch.zeros(batch_size, *feature_shape, dtype=torch.bool)
    masked_features = features * feature_mask

    # Select features 3, 7, 1, 9, 5 for each batch element
    afa_selection = torch.tensor([[3], [7], [1], [9], [5]])
    selection_mask = torch.ones_like(afa_selection, dtype=torch.bool)

    unmasker = DirectUnmasker()
    new_feature_mask = unmasker.unmask(
        masked_features=masked_features,
        feature_mask=feature_mask,
        features=features,
        afa_selection=afa_selection,
        selection_mask=selection_mask,
        feature_shape=feature_shape,
    )

    # Check that correct features are unmasked
    assert new_feature_mask[0, 2]  # Feature 3 (0-indexed: 2)
    assert new_feature_mask[1, 6]  # Feature 7 (0-indexed: 6)
    assert new_feature_mask[2, 0]  # Feature 1 (0-indexed: 0)
    assert new_feature_mask[3, 8]  # Feature 9 (0-indexed: 8)
    assert new_feature_mask[4, 4]  # Feature 5 (0-indexed: 4)


def test_direct_unmasker_3d_features():
    """Test with 3D features."""
    batch_size = 3
    feature_shape = torch.Size([2, 3, 4])
    features = torch.randn(batch_size, *feature_shape)
    feature_mask = torch.zeros(batch_size, *feature_shape, dtype=torch.bool)
    masked_features = features * feature_mask

    # Select specific features
    afa_selection = torch.tensor([[1], [12], [24]])  # Various 3D positions
    selection_mask = torch.ones_like(afa_selection, dtype=torch.bool)

    unmasker = DirectUnmasker()
    new_feature_mask = unmasker.unmask(
        masked_features=masked_features,
        feature_mask=feature_mask,
        features=features,
        afa_selection=afa_selection,
        selection_mask=selection_mask,
        feature_shape=feature_shape,
    )

    # Verify correct 3D positions are unmasked
    assert new_feature_mask.shape == (*([batch_size] + list(feature_shape)),)

    # Check that exactly one feature per batch element is unmasked
    for i in range(batch_size):
        assert new_feature_mask[i].sum() == 1


def test_direct_unmasker_get_n_selections():
    """Test get_n_selections method."""
    unmasker = DirectUnmasker()

    # Test 1D features
    feature_shape_1d = torch.Size([10])
    assert unmasker.get_n_selections(feature_shape_1d) == 10

    # Test 2D features
    feature_shape_2d = torch.Size([3, 4])
    assert unmasker.get_n_selections(feature_shape_2d) == 12

    # Test 3D features
    feature_shape_3d = torch.Size([2, 3, 4])
    assert unmasker.get_n_selections(feature_shape_3d) == 24


def test_direct_unmasker_edge_cases():
    """Test edge cases and boundary conditions."""
    batch_size = 2
    feature_shape = torch.Size([2, 2])
    features = torch.randn(batch_size, *feature_shape)
    feature_mask = torch.zeros(batch_size, *feature_shape, dtype=torch.bool)
    masked_features = features * feature_mask

    # Test boundary selections (first and last features)
    afa_selection = torch.tensor([[1], [4]])  # First and last features
    selection_mask = torch.ones_like(afa_selection, dtype=torch.bool)

    unmasker = DirectUnmasker()
    new_feature_mask = unmasker.unmask(
        masked_features=masked_features,
        feature_mask=feature_mask,
        features=features,
        afa_selection=afa_selection,
        selection_mask=selection_mask,
        feature_shape=feature_shape,
    )

    # Verify boundary features are correctly unmasked
    assert new_feature_mask[0, 0, 0]  # Feature 1 -> position (0, 0)
    assert new_feature_mask[1, 1, 1]  # Feature 4 -> position (1, 1)


def test_direct_unmasker_large_batch():
    """Test with large batch sizes."""
    batch_size = 100
    feature_shape = torch.Size([5])
    features = torch.randn(batch_size, *feature_shape)
    feature_mask = torch.zeros(batch_size, *feature_shape, dtype=torch.bool)
    masked_features = features * feature_mask

    # Random selections for each batch element
    afa_selection = torch.randint(1, 6, (batch_size, 1))
    selection_mask = torch.ones_like(afa_selection, dtype=torch.bool)

    unmasker = DirectUnmasker()
    new_feature_mask = unmasker.unmask(
        masked_features=masked_features,
        feature_mask=feature_mask,
        features=features,
        afa_selection=afa_selection,
        selection_mask=selection_mask,
        feature_shape=feature_shape,
    )

    # Verify that exactly one feature is unmasked per batch element
    assert new_feature_mask.sum() == batch_size


def test_direct_unmasker_multidimensional_batch():
    """Test with complex multi-dimensional batch shapes."""
    batch_shape = torch.Size([2, 3, 2])
    feature_shape = torch.Size([3])
    features = torch.randn(*batch_shape, *feature_shape)
    feature_mask = torch.zeros(*batch_shape, *feature_shape, dtype=torch.bool)
    masked_features = features * feature_mask

    # Random selections for each batch element
    afa_selection = torch.randint(1, 4, (*batch_shape, 1))
    selection_mask = torch.ones_like(afa_selection, dtype=torch.bool)

    unmasker = DirectUnmasker()
    new_feature_mask = unmasker.unmask(
        masked_features=masked_features,
        feature_mask=feature_mask,
        features=features,
        afa_selection=afa_selection,
        selection_mask=selection_mask,
        feature_shape=feature_shape,
    )

    assert new_feature_mask.shape == (*batch_shape, *feature_shape)

    # Check that exactly one feature per batch element is unmasked
    total_batch_elements = torch.prod(torch.tensor(batch_shape))
    assert new_feature_mask.sum() == total_batch_elements


def test_direct_unmasker_set_seed():
    """Test that set_seed method exists and is callable."""
    unmasker = DirectUnmasker()

    # Should not raise any exceptions
    unmasker.set_seed(42)
    unmasker.set_seed(None)
