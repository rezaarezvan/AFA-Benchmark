from typing import ClassVar, final, override

import numpy as np  # Keep for mutual_info_classif, will remove other numpy usage
import torch
from sklearn.feature_selection import mutual_info_classif

from afabench.common.config_classes import (
    AACODefaultInitializerConfig,
    FixedRandomInitializerConfig,
    LeastInformativeInitializerConfig,
    ManualInitializerConfig,
    MutualInformationInitializerConfig,
    RandomPerEpisodeInitializerConfig,
)
from afabench.common.custom_types import (
    AFAInitializer,
    FeatureMask,
    Features,
    Label,
)


@final
class FixedRandomInitializer(AFAInitializer):
    """Select random features once and reuse them across all episodes."""

    def __init__(self, config: FixedRandomInitializerConfig):
        self.config = config
        self._cached_mask: FeatureMask | None = None
        self.rng = torch.Generator()

    @override
    def set_seed(self, seed: int | None) -> None:
        if seed is not None:
            self.rng.manual_seed(seed)
        else:
            # Re-initialize with a new random seed if None is provided
            self.rng = torch.Generator()

    @override
    def initialize(
        self,
        features: Features,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> FeatureMask:
        assert feature_shape is not None, (
            "feature_shape must be provided for FixedRandomInitializer"
        )
        # We can figure out the batch shape by subtracting the feature shape
        batch_shape = features.shape[: -len(feature_shape)]
        # Select once and cache
        if self._cached_mask is None:
            assert feature_shape is not None, (
                "feature_shape must be provided for FixedRandomInitializer"
            )
            num_features = feature_shape.numel()
            num_features_to_unmask = int(
                num_features * self.config.unmask_ratio
            )
            # Sample feature indices
            flat_indices = torch.randperm(num_features, generator=self.rng)[
                :num_features_to_unmask
            ]
            # Unflatten
            multi_indices = torch.unravel_index(flat_indices, feature_shape)
            # Turn into mask
            self._cached_mask = torch.zeros(feature_shape, dtype=torch.bool)
            self._cached_mask[multi_indices] = True

        assert self._cached_mask is not None
        # Expand the mask to match batch dimensions
        # Create a copy and add singleton batch dimensions
        expanded_mask = self._cached_mask
        for _ in range(len(batch_shape)):
            expanded_mask = expanded_mask.unsqueeze(0)
        return expanded_mask.expand(batch_shape + feature_shape)


@final
class DynamicRandomInitializer(AFAInitializer):
    """Select different random features for each episode."""

    def __init__(self, config: RandomPerEpisodeInitializerConfig):
        self.config = config
        self.rng = torch.Generator()

    @override
    def set_seed(self, seed: int | None) -> None:
        if seed is not None:
            self.rng.manual_seed(seed)
        else:
            self.rng = torch.Generator()

    @override
    def initialize(
        self,
        features: Features,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> FeatureMask:
        assert feature_shape is not None, (
            "feature_shape must be provided for DynamicRandomInitializer"
        )
        # We can figure out the batch shape by subtracting the feature shape
        batch_shape = features.shape[: -len(feature_shape)]
        batch_size = int(torch.prod(torch.tensor(batch_shape)))

        num_features = feature_shape.numel()
        num_features_to_unmask = int(num_features * self.config.unmask_ratio)

        # Create different random masks for each batch element
        masks = []
        for _ in range(batch_size):
            flat_indices = torch.randperm(num_features, generator=self.rng)[
                :num_features_to_unmask
            ]
            multi_indices = torch.unravel_index(flat_indices, feature_shape)
            mask = torch.zeros(feature_shape, dtype=torch.bool)
            mask[multi_indices] = True
            masks.append(mask)

        # Stack masks and reshape to match batch dimensions
        stacked_masks = torch.stack(masks, dim=0)
        return stacked_masks.view(batch_shape + feature_shape)


@final
class ManualInitializer(AFAInitializer):
    """Use explicitly specified feature indices."""

    def __init__(self, config: ManualInitializerConfig):
        self.flat_feature_indices = torch.tensor(
            config.flat_feature_indices, dtype=torch.long
        )

    @override
    def set_seed(self, seed: int | None) -> None:
        # Manual initializer is deterministic, so seed has no effect
        pass

    @override
    def initialize(
        self,
        features: Features,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> FeatureMask:
        assert feature_shape is not None, (
            "feature_shape must be provided for ManualInitializer"
        )
        # We can figure out the batch shape by subtracting the feature shape
        batch_shape = features.shape[: -len(feature_shape)]
        num_features_to_unmask = self.flat_feature_indices.numel()
        num_features = feature_shape.numel()

        # Ensure feature indices are valid
        assert (self.flat_feature_indices >= 0).all() and (  # noqa: PT018
            self.flat_feature_indices < num_features
        ).all(), "Feature index out of bounds"

        mask = torch.zeros(feature_shape, dtype=torch.bool)
        multi_indices = torch.unravel_index(
            self.flat_feature_indices, feature_shape
        )
        mask[multi_indices] = True

        assert mask.sum() == num_features_to_unmask, f"Expected {
            num_features_to_unmask
        } features, got {mask.sum()}"

        # Expand to match batch dimensions
        # Add singleton batch dimensions and expand
        for _ in range(len(batch_shape)):
            mask = mask.unsqueeze(0)
        return mask.expand(batch_shape + feature_shape)

    @override
    def __repr__(self) -> str:
        return (
            f"ManualInitializer(features={self.flat_feature_indices.tolist()})"
        )


@final
class MutualInformationInitializer(AFAInitializer):
    """Select features with highest mutual information with target labels."""

    def __init__(self, config: MutualInformationInitializerConfig):
        self.config = config
        self._cached_ranking: np.ndarray | None = None  # pyright: ignore[reportMissingTypeArgument]
        self._seed: int | None = None

    @override
    def set_seed(self, seed: int | None) -> None:
        self._seed = seed
        self._cached_ranking = None  # Clear cache if seed changes

    @override
    def initialize(
        self,
        features: Features,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> FeatureMask:
        assert features is not None, (
            f"{self.__class__.__name__} requires features"
        )
        assert label is not None, f"{self.__class__.__name__} requires label"
        assert feature_shape is not None, (
            "feature_shape must be provided for MutualInformationInitializer"
        )

        num_features = feature_shape.numel()
        num_features_to_unmask = int(num_features * self.config.unmask_ratio)

        # We can figure out the batch shape by subtracting the feature shape
        batch_shape = features.shape[: -len(feature_shape)]

        # Flatten batch dimensions for sklearn
        features_flat = features.view(-1, *feature_shape)
        label_flat = label.view(-1)

        # Convert to numpy for mutual_info_classif
        train_features_np = features_flat.cpu().numpy()
        train_labels_np = label_flat.cpu().numpy()

        # Compute MI once and cache ranking (NumPy array)
        if self._cached_ranking is None:
            # Reshape features to 2D for mutual_info_classif if necessary
            if train_features_np.ndim > 2:
                train_features_flat = train_features_np.reshape(
                    train_features_np.shape[0], -1
                )
            else:
                train_features_flat = train_features_np

            mi_scores = mutual_info_classif(
                train_features_flat.copy(),
                train_labels_np,
                random_state=self._seed,
            )
            # Sort descending by MI score
            self._cached_ranking = np.argsort(mi_scores)[::-1]

        # Get top features (still a NumPy array)
        selected_flat_indices_np = self._cached_ranking[
            :num_features_to_unmask
        ]

        # Convert to PyTorch tensor for unraveling
        selected_flat_indices = torch.tensor(
            selected_flat_indices_np.copy(), dtype=torch.long
        )

        # We can figure out the batch shape by subtracting the feature shape
        batch_shape = features.shape[: -len(feature_shape)]

        # Create mask
        mask = torch.zeros(feature_shape, dtype=torch.bool)
        multi_indices = torch.unravel_index(
            selected_flat_indices, feature_shape
        )
        mask[multi_indices] = True

        # Expand to match batch dimensions
        # Add singleton batch dimensions and expand
        for _ in range(len(batch_shape)):
            mask = mask.unsqueeze(0)
        return mask.expand(batch_shape + feature_shape)


@final
class LeastInformativeInitializer(AFAInitializer):
    """
    Select features with LOWEST mutual information (adversarial baseline).

    Useful for robustness testing - how well do methods perform with poor initialization?
    """

    def __init__(self, config: LeastInformativeInitializerConfig):
        self.config = config
        self._cached_ranking: np.ndarray | None = None  # pyright: ignore[reportMissingTypeArgument]
        self._seed: int | None = None

    @override
    def set_seed(self, seed: int | None) -> None:
        self._seed = seed
        self._cached_ranking = None  # Clear cache if seed changes

    @override
    def initialize(
        self,
        features: Features,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> FeatureMask:
        assert features is not None, (
            f"{self.__class__.__name__} requires features"
        )
        assert label is not None, f"{self.__class__.__name__} requires label"
        assert feature_shape is not None, (
            "feature_shape must be provided for LeastInformativeInitializer"
        )

        num_features = feature_shape.numel()
        num_features_to_unmask = int(num_features * self.config.unmask_ratio)

        # We can figure out the batch shape by subtracting the feature shape
        batch_shape = features.shape[: -len(feature_shape)]

        # Flatten batch dimensions for sklearn
        features_flat = features.view(-1, *feature_shape)
        label_flat = label.view(-1)

        # Convert to numpy for mutual_info_classif
        train_features_np = features_flat.cpu().numpy()
        train_labels_np = label_flat.cpu().numpy()

        # Compute MI once and cache ranking (NumPy array)
        if self._cached_ranking is None:
            # Reshape features to 2D for mutual_info_classif if necessary
            if train_features_np.ndim > 2:
                train_features_flat = train_features_np.reshape(
                    train_features_np.shape[0], -1
                )
            else:
                train_features_flat = train_features_np

            mi_scores = mutual_info_classif(
                train_features_flat, train_labels_np, random_state=self._seed
            )
            # Sort ascending by MI score (worst first)
            self._cached_ranking = np.argsort(mi_scores)

        selected_flat_indices_np = self._cached_ranking[
            :num_features_to_unmask
        ]

        # Convert to PyTorch tensor for unraveling
        selected_flat_indices = torch.tensor(
            selected_flat_indices_np, dtype=torch.long
        )

        # We can figure out the batch shape by subtracting the feature shape
        batch_shape = features.shape[: -len(feature_shape)]

        # Create mask
        mask = torch.zeros(feature_shape, dtype=torch.bool)
        multi_indices = torch.unravel_index(
            selected_flat_indices, feature_shape
        )
        mask[multi_indices] = True

        # Expand to match batch dimensions
        # Add singleton batch dimensions and expand
        for _ in range(len(batch_shape)):
            mask = mask.unsqueeze(0)
        return mask.expand(batch_shape + feature_shape)


@final
class ZeroInitializer(AFAInitializer):
    """Cold-start initializer that selects no features."""

    def __init__(self):
        pass

    @override
    def set_seed(self, seed: int | None) -> None:
        # Zero initializer is deterministic, so seed has no effect
        pass

    @override
    def initialize(
        self,
        features: Features,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> FeatureMask:
        assert feature_shape is not None, (
            "feature_shape must be provided for ZeroInitializer"
        )
        # We can figure out the batch shape by subtracting the feature shape
        batch_shape = features.shape[: -len(feature_shape)]

        # Select no features (all False mask) and expand to batch dimensions
        mask = torch.zeros(feature_shape, dtype=torch.bool)
        # Add singleton batch dimensions and expand
        for _ in range(len(batch_shape)):
            mask = mask.unsqueeze(0)
        return mask.expand(batch_shape + feature_shape)


@final
class AACODefaultInitializer(AFAInitializer):
    """
    AACO paper's deterministic initial feature selection.

    Uses dataset-specific initial features as per the original paper:
    - cube: feature 6
    - mnist/fashionmnist: feature 100
    - others: middle feature
    """

    DATASET_INITIAL_FEATURES: ClassVar = {
        "cube": 6,
        "cubesimple": 3,
        "grid": 1,
        "gas10": 6,
        "mnist": 100,
        "fashionmnist": 100,
        "afacontext": 0,  # context feature first
    }

    def __init__(self, config: AACODefaultInitializerConfig):
        self.dataset_name = config.dataset_name.lower()

    @override
    def set_seed(self, seed: int | None) -> None:
        # AACODefaultInitializer is deterministic, so seed has no effect
        pass

    @override
    def initialize(
        self,
        features: Features,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> FeatureMask:
        assert feature_shape is not None, (
            "feature_shape must be provided for AACODefaultInitializer"
        )
        num_features = feature_shape.numel()

        initial_feature_flat_index = self.DATASET_INITIAL_FEATURES.get(
            self.dataset_name,
            num_features // 2,  # fallback: middle feature
        )

        # AACO always starts with exactly 1 feature
        # This implicitly means the intended unmask_ratio leads to 1 feature
        # The ManualInitializer has a similar check for consistency.

        # Ensure the selected feature is within bounds
        assert 0 <= initial_feature_flat_index < num_features, (
            f"Initial feature index {initial_feature_flat_index} for dataset "
            f"'{self.dataset_name}' is out of bounds for {num_features} features."
        )

        # We can figure out the batch shape by subtracting the feature shape
        batch_shape = features.shape[: -len(feature_shape)]

        mask = torch.zeros(feature_shape, dtype=torch.bool)
        initial_feature_flat_index_tensor = torch.tensor(
            initial_feature_flat_index, dtype=torch.long
        )
        multi_indices = torch.unravel_index(
            initial_feature_flat_index_tensor, feature_shape
        )
        # multi_indices will be a tuple of arrays, e.g., (array([x]), array([y]), ...).
        # We need to ensure it's applied correctly for scalar indices too by directly passing the tuple.
        mask[multi_indices] = True

        # Expand to match batch dimensions
        # Add singleton batch dimensions and expand
        for _ in range(len(batch_shape)):
            mask = mask.unsqueeze(0)
        return mask.expand(batch_shape + feature_shape)
