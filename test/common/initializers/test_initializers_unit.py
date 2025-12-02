import pytest
import torch

from afabench.common.afa_initializers import (
    AACODefaultInitializer,
    FixedRandomInitializer,
    MutualInformationInitializer,
    ZeroInitializer,
)
from afabench.common.config_classes import (
    AACODefaultInitializerConfig,
    FixedRandomInitializerConfig,
    MutualInformationInitializerConfig,
)
from afabench.common.custom_types import Features, Label


# Fixtures
@pytest.fixture
def fixture() -> tuple[Features, Label, torch.Size]:
    """Synthetic 2D data for testing (torch)."""
    n_samples = 100
    feature_shape = torch.Size([2, 5])  # 2D features
    x_torch: Features = torch.randn(n_samples, *feature_shape).float()
    y_torch: Label = (x_torch[:, 0, 0] + x_torch[:, 0, 1] > 0).long()
    return x_torch, y_torch, feature_shape


# Test FixedRandomInitializer


def test_fixed_random_feature_shape(
    fixture: tuple[Features, Label, torch.Size],
) -> None:
    """Fixed random should work with the given feature shape."""
    x, _, feature_shape = fixture
    config = FixedRandomInitializerConfig(unmask_ratio=0.3)
    initializer = FixedRandomInitializer(config)
    initializer.set_seed(42)

    mask = initializer.initialize(
        features=x,
        feature_shape=feature_shape,
    )
    assert mask.shape == feature_shape
    assert mask.sum() == int(feature_shape.numel() * 0.3)


# Test RandomPerEpisodeInitializer
# TODO:


# Test ManualInitializer
# TODO:


# Test MutualInformationInitializer
# TODO:


def test_mutual_information_caching(
    fixture: tuple[Features, Label, torch.Size],
) -> None:
    """MI initializer should cache results."""
    x, y, feature_shape = fixture

    config = MutualInformationInitializerConfig(unmask_ratio=0.3)
    initializer = MutualInformationInitializer(config)
    initializer.set_seed(42)

    mask1 = initializer.initialize(
        features=x, label=y, feature_shape=feature_shape
    )
    mask2 = initializer.initialize(
        features=x, label=y, feature_shape=feature_shape
    )
    assert torch.equal(mask1, mask2)


def test_mutual_information_torch_2d_features(
    fixture: tuple[Features, Label, torch.Size],
) -> None:
    """MI initializer should work with 2D features and reshape internally."""
    x, y, feature_shape = fixture

    config = MutualInformationInitializerConfig(unmask_ratio=0.2)
    initializer = MutualInformationInitializer(config)
    initializer.set_seed(42)

    mask = initializer.initialize(
        features=x, label=y, feature_shape=feature_shape
    )

    assert mask.shape == feature_shape
    assert mask.sum() == int(feature_shape.numel() * 0.2)

    # Get multi-indices, convert to flat for comparison
    multi_indices = mask.nonzero(as_tuple=True)
    selected_flat_indices = (
        multi_indices[0] * feature_shape[1] + multi_indices[1]
    ).tolist()

    assert 0 in selected_flat_indices
    assert 1 in selected_flat_indices


# Test LeastInformativeInitializer
# TODO:


# Test ZeroInitializer
# TODO:


def test_zero_initializer_feature_shape(
    fixture: tuple[Features, Label, torch.Size],
) -> None:
    """Zero initializer should return an empty mask for the given feature shape."""
    x, _, feature_shape = fixture
    initializer = ZeroInitializer()
    mask = initializer.initialize(
        features=x,
        feature_shape=feature_shape,
    )
    assert mask.shape == feature_shape
    assert mask.sum() == 0


# Test AACODefaultInitializer
# TODO:


def test_aaco_default_initializer_feature_shape(
    fixture: tuple[Features, Label, torch.Size],
) -> None:
    """AACO initializer should work with the given feature shape, selecting flattened index."""
    x, _, feature_shape = fixture
    config = AACODefaultInitializerConfig(
        dataset_name="cubesimple"
    )  # Feature 3
    initializer = AACODefaultInitializer(config)

    mask = initializer.initialize(
        features=x,
        feature_shape=feature_shape,
    )

    expected_mask = torch.zeros(feature_shape, dtype=torch.bool)
    flat_index = 3  # For cubesimple
    multi_index = torch.unravel_index(
        torch.tensor(flat_index, dtype=torch.long), feature_shape
    )
    expected_mask[multi_index] = True

    assert torch.equal(mask, expected_mask)
    assert mask.sum() == 1


# Integration tests
