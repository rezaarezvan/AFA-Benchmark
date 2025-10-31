import torch
import pytest
import numpy as np

from common.warm_start import (
    FixedRandomStrategy,
    LeastInformativeStrategy,
    ManualStrategy,
    MutualInformationStrategy,
    RandomPerEpisodeStrategy,
)


# Fixtures
@pytest.fixture
def simple_data():
    """Simple synthetic data for testing."""
    n_samples, n_features = 100, 10
    # Features 0, 1 are informative, rest are noise
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture
def torch_data():
    """Torch tensor data."""
    n_samples, n_features = 100, 10
    X = torch.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).long()
    return X, y


# Test FixedRandomStrategy
def test_fixed_random_deterministic():
    """Fixed random should return same features across calls."""
    strategy = FixedRandomStrategy(seed=42)
    features1 = strategy.select_features(10, 3)
    features2 = strategy.select_features(10, 3)
    assert features1 == features2
    assert len(features1) == 3


def test_fixed_random_different_seeds():
    """Different seeds should give different features."""
    strategy1 = FixedRandomStrategy(seed=42)
    strategy2 = FixedRandomStrategy(seed=43)
    features1 = strategy1.select_features(10, 3)
    features2 = strategy2.select_features(10, 3)
    assert features1 != features2


# Test RandomPerEpisodeStrategy
def test_random_per_episode_varies():
    """Random per episode should give different results."""
    strategy = RandomPerEpisodeStrategy(seed=42)
    features1 = strategy.select_features(10, 3)
    features2 = strategy.select_features(10, 3)
    assert features1 != features2
    assert len(features1) == 3
    assert len(features2) == 3


# Test ManualStrategy
def test_manual_strategy():
    """Manual strategy should return exact indices."""
    indices = [0, 5, 9]
    strategy = ManualStrategy(feature_indices=indices)
    features = strategy.select_features(10, 3)
    assert features == indices


def test_manual_strategy_wrong_count():
    """Manual strategy should raise error if count mismatch."""
    indices = [0, 5]
    strategy = ManualStrategy(feature_indices=indices)
    with pytest.raises(AssertionError):
        strategy.select_features(10, 3)


# Test MutualInformationStrategy
def test_mutual_information_no_data():
    """MI strategy should raise error without data."""
    strategy = MutualInformationStrategy(seed=42)
    with pytest.raises(AssertionError):
        strategy.select_features(10, 3)


def test_mutual_information_with_data(simple_data):
    """MI strategy should select most informative features."""
    X, y = simple_data
    strategy = MutualInformationStrategy(seed=42)
    features = strategy.select_features(
        10, 2, train_features=X, train_labels=y
    )

    # Should select features 0 and 1 (most informative in our data)
    assert len(features) == 2
    assert 0 in features or 1 in features  # At least one should be selected


def test_mutual_information_caching(simple_data):
    """MI strategy should cache results."""
    X, y = simple_data
    strategy = MutualInformationStrategy(seed=42)
    features1 = strategy.select_features(
        10, 3, train_features=X, train_labels=y
    )
    features2 = strategy.select_features(
        10, 3, train_features=X, train_labels=y
    )
    assert features1 == features2


def test_mutual_information_torch(torch_data):
    """MI strategy should work with torch tensors."""
    X, y = torch_data
    strategy = MutualInformationStrategy(seed=42)
    features = strategy.select_features(
        10, 3, train_features=X, train_labels=y
    )
    assert len(features) == 3


# Test LeastInformativeStrategy
def test_least_informative_with_data(simple_data):
    """Least informative should select worst features."""
    X, y = simple_data
    strategy = LeastInformativeStrategy(seed=42)
    features = strategy.select_features(
        10, 3, train_features=X, train_labels=y
    )

    assert len(features) == 3
    assert all(f not in [0, 1] for f in features)


# Integration tests
def test_no_duplicates():
    """Selected features should not have duplicates."""
    strategy = FixedRandomStrategy(seed=42)
    features = strategy.select_features(20, 5)
    assert len(features) == len(set(features))


def test_valid_indices():
    """Selected features should be valid indices."""
    strategy = FixedRandomStrategy(seed=42)
    features = strategy.select_features(10, 5)
    assert all(0 <= f < 10 for f in features)
