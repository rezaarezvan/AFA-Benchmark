"""Tests for SyntheticMNISTDataset properties and behavior."""

import pytest
import torch

from afabench.common.datasets.datasets import SyntheticMNISTDataset


@pytest.fixture
def small_synthetic_mnist() -> SyntheticMNISTDataset:
    """Create a small synthetic MNIST dataset for testing."""
    return SyntheticMNISTDataset(seed=42, n_samples=100)


@pytest.fixture
def large_synthetic_mnist() -> SyntheticMNISTDataset:
    """Create a larger synthetic MNIST dataset for testing."""
    return SyntheticMNISTDataset(seed=123, n_samples=1000)


def test_synthetic_mnist_feature_shape(
    small_synthetic_mnist: SyntheticMNISTDataset,
) -> None:
    """Test that synthetic MNIST has correct feature shape."""
    assert small_synthetic_mnist.feature_shape == torch.Size([1, 28, 28])


def test_synthetic_mnist_label_shape(
    small_synthetic_mnist: SyntheticMNISTDataset,
) -> None:
    """Test that synthetic MNIST has correct label shape."""
    assert small_synthetic_mnist.label_shape == torch.Size([10])


def test_synthetic_mnist_accepts_seed() -> None:
    """Test that synthetic MNIST accepts seed parameter."""
    assert SyntheticMNISTDataset.accepts_seed()


def test_synthetic_mnist_data_properties(
    small_synthetic_mnist: SyntheticMNISTDataset,
) -> None:
    """Test that synthetic MNIST data has expected properties."""
    features, labels = small_synthetic_mnist.get_all_data()

    # Check overall shapes
    assert features.shape == (100, 1, 28, 28)
    assert labels.shape == (100, 10)

    # Check feature range [0, 1]
    assert torch.all(features >= 0)
    assert torch.all(features <= 1)

    # Check labels are one-hot encoded
    label_sums = torch.sum(labels, dim=1)
    assert torch.allclose(label_sums, torch.ones_like(label_sums))

    # Check that each label vector has exactly one 1.0
    for i in range(labels.shape[0]):
        assert torch.sum(labels[i] == 1.0) == 1


def test_synthetic_mnist_data_format_conversion(
    small_synthetic_mnist: SyntheticMNISTDataset,
) -> None:
    """Test that features can be converted between image and flattened format."""
    features, labels = small_synthetic_mnist.get_all_data()

    # Test flattening to 784 dimensions
    flattened = features.view(-1, 784)
    assert flattened.shape == (features.shape[0], 784)

    # Test reshaping back to image format
    reshaped_back = flattened.view(-1, 1, 28, 28)
    assert reshaped_back.shape == features.shape

    # Check that conversion preserves data
    assert torch.allclose(reshaped_back, features)


def test_synthetic_mnist_seed_reproducibility() -> None:
    """Test that the same seed produces identical data."""
    dataset1 = SyntheticMNISTDataset(seed=42, n_samples=50)
    dataset2 = SyntheticMNISTDataset(seed=42, n_samples=50)

    features1, labels1 = dataset1.get_all_data()
    features2, labels2 = dataset2.get_all_data()

    assert torch.allclose(features1, features2)
    assert torch.allclose(labels1, labels2)


def test_synthetic_mnist_different_seeds_produce_different_data() -> None:
    """Test that different seeds produce different data."""
    dataset1 = SyntheticMNISTDataset(seed=42, n_samples=50)
    dataset2 = SyntheticMNISTDataset(seed=123, n_samples=50)

    features1, labels1 = dataset1.get_all_data()
    features2, labels2 = dataset2.get_all_data()

    # Features should be different
    assert not torch.allclose(features1, features2)
    # But shapes should be the same
    assert features1.shape == features2.shape
    assert labels1.shape == labels2.shape


def test_synthetic_mnist_class_distribution(
    large_synthetic_mnist: SyntheticMNISTDataset,
) -> None:
    """Test that all classes are represented in the dataset."""
    features, labels = large_synthetic_mnist.get_all_data()

    # Convert one-hot to class indices
    class_indices = torch.argmax(labels, dim=1)

    # Count occurrences of each class
    class_counts = torch.bincount(class_indices, minlength=10)

    # All 10 classes should be present
    assert len(class_counts) == 10
    assert torch.all(class_counts > 0)

    # Distribution should be roughly uniform (within reasonable bounds)
    expected_count = len(large_synthetic_mnist) // 10
    for count in class_counts:
        # Allow up to 50% deviation from perfect uniformity
        assert abs(count.item() - expected_count) < expected_count * 0.5


def test_synthetic_mnist_patterns_are_learnable(
    small_synthetic_mnist: SyntheticMNISTDataset,
) -> None:
    """Test that different classes have distinguishable patterns."""
    features, labels = small_synthetic_mnist.get_all_data()

    # Convert one-hot to class indices
    class_indices = torch.argmax(labels, dim=1)

    # Get samples for first few classes
    class_features = {}
    for class_idx in range(min(5, 10)):  # Test first 5 classes
        mask = class_indices == class_idx
        if torch.any(mask):
            class_features[class_idx] = features[mask][
                0
            ]  # First sample of this class

    # Classes should have different feature patterns
    if len(class_features) >= 2:
        classes = list(class_features.keys())
        feat1 = class_features[classes[0]]
        feat2 = class_features[classes[1]]

        # Features should not be identical (patterns should differ)
        assert not torch.allclose(feat1, feat2, atol=1e-4)


def test_synthetic_mnist_getitem(
    small_synthetic_mnist: SyntheticMNISTDataset,
) -> None:
    """Test __getitem__ method returns correct shapes."""
    features, labels = small_synthetic_mnist[0]

    assert features.shape == torch.Size([1, 28, 28])
    assert labels.shape == torch.Size([10])
    assert torch.sum(labels) == 1.0  # One-hot encoded


def test_synthetic_mnist_len(
    small_synthetic_mnist: SyntheticMNISTDataset,
) -> None:
    """Test __len__ method returns correct length."""
    assert len(small_synthetic_mnist) == 100


def test_synthetic_mnist_custom_parameters() -> None:
    """Test synthetic MNIST with custom parameters."""
    dataset = SyntheticMNISTDataset(
        seed=999, n_samples=200, noise_std=0.2, pattern_intensity=0.5
    )

    assert len(dataset) == 200
    features, labels = dataset.get_all_data()
    assert features.shape == (200, 1, 28, 28)
    assert labels.shape == (200, 10)


def test_synthetic_mnist_subset_creation(
    small_synthetic_mnist: SyntheticMNISTDataset,
) -> None:
    """Test that subset creation works correctly."""
    indices = [0, 10, 20, 30, 40]
    subset = small_synthetic_mnist.create_subset(indices)

    assert len(subset) == 5

    # Test that subset contains correct data
    original_features, _ = small_synthetic_mnist.get_all_data()
    subset_features, _ = subset.get_all_data()

    for i, idx in enumerate(indices):
        assert torch.allclose(subset_features[i], original_features[idx])


def test_synthetic_mnist_feature_intensity_ranges(
    small_synthetic_mnist: SyntheticMNISTDataset,
) -> None:
    """Test that generated features have reasonable intensity distributions."""
    features, labels = small_synthetic_mnist.get_all_data()

    # Features should use the full [0, 1] range (not just background noise)
    assert torch.min(features) >= 0.0
    assert torch.max(features) <= 1.0

    # Should have some high-intensity pixels from patterns
    assert torch.max(features) > 0.5

    # Should have some low-intensity pixels from background
    assert torch.min(features) < 0.5


def test_synthetic_mnist_pattern_consistency() -> None:
    """Test that the same seed and same n_samples produces identical patterns."""
    dataset1 = SyntheticMNISTDataset(seed=42, n_samples=100)
    dataset2 = SyntheticMNISTDataset(
        seed=42, n_samples=100
    )  # Same parameters, should be identical

    features1, labels1 = dataset1.get_all_data()
    features2, labels2 = dataset2.get_all_data()

    # Both datasets should be identical
    assert torch.allclose(features1, features2)
    assert torch.allclose(labels1, labels2)
