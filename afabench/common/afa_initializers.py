from typing import final, override

import numpy as np
import torch
from sklearn.feature_selection import mutual_info_classif

from afabench.common.config_classes import (
    AACODefaultInitializerConfig,
    FixedRandomInitializerConfig,
    ManualInitializerConfig,
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
        self.rng = np.random.default_rng(None)

    @override
    def set_seed(self, seed: int | None) -> None:
        self.rng = np.random.default_rng(seed)

    @override
    def initialize(
        self,
        features: Features,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> FeatureMask:
        # Select once and cache
        if self._cached_mask is None:
            assert feature_shape is not None, (
                "feature_shape must be provided for FixedRandomInitializer"
            )
            num_features = int(np.prod(list(feature_shape)))
            num_features_to_unmask = num_features * self.config.unmask_ratio
            # Sample feature indices
            flat_indices = self.rng.choice(
                num_features,
                size=int(num_features_to_unmask),
                replace=False,
            )
            # Unflatten
            multi_indices = np.unravel_index(flat_indices, feature_shape)
            # Turn into mask
            self._cached_mask = torch.zeros(feature_shape, dtype=torch.bool)
            self._cached_mask[multi_indices] = True

        assert self._cached_mask is not None
        return self._cached_mask


class RandomPerEpisodeInitializer(AFAInitializer):
    """
    Select different random features for each episode.
    """

    def select_features(
        self,
        n_features_total: int,
        n_features_select: int,
        train_features: Features | None = None,
        train_labels: Label | None = None,
    ) -> list[int]:
        # Select new features each time
        return self.rng.choice(
            n_features_total, size=n_features_select, replace=False
        ).tolist()


@final
class ManualInitializer(AFAInitializer):
    """Use explicitly specified feature indices."""

    def __init__(self, config: ManualInitializerConfig):
        super().__init__()
        self.feature_indices = config.feature_indices

    def select_features(
        self,
        n_features_total: int,
        n_features_select: int,
        train_features: Features | None = None,
        train_labels: Label | None = None,
    ) -> list[int]:
        assert len(self.feature_indices) == n_features_select, f"Expected {
            n_features_select
        } features, got {len(self.feature_indices)}"
        return self.feature_indices

    def __repr__(self) -> str:
        return f"ManualStrategy(features={self.feature_indices})"


class MutualInformationInitializer(AFAInitializer):
    """
    Select features with highest mutual information with target labels.
    """

    def __init__(self, seed: int | None = None):
        super().__init__(seed)
        self._cached_ranking: np.ndarray | None = None

    def select_features(
        self,
        n_features_total: int,
        n_features_select: int,
        train_features: Features | None = None,
        train_labels: Label | None = None,
    ) -> list[int]:
        assert train_features is not None and train_labels is not None, f"{
            self.__class__.__name__
        } requires train_features and train_labels"

        # Convert to numpy if needed
        train_features = (
            train_features.numpy()
            if isinstance(train_features, torch.Tensor)
            else train_features
        )
        train_labels = (
            train_labels.numpy()
            if isinstance(train_labels, torch.Tensor)
            else train_labels
        )

        # Compute MI once and cache ranking
        if self._cached_ranking is None:
            mi_scores = mutual_info_classif(
                train_features, train_labels, random_state=self.seed
            )
            # Sort descending by MI score
            self._cached_ranking = np.argsort(mi_scores)[::-1]

        return self._cached_ranking[:n_features_select].tolist()


class LeastInformativeInitializer(AFAInitializer):
    """
    Select features with LOWEST mutual information (adversarial baseline).

    Useful for robustness testing - how well do methods perform with poor initialization?
    """

    def __init__(self, seed: int | None = None):
        super().__init__(seed)
        self._cached_ranking: np.ndarray | None = None

    def select_features(
        self,
        n_features_total: int,
        n_features_select: int,
        train_features: Features | None = None,
        train_labels: Label | None = None,
    ) -> list[int]:
        assert train_features is not None and train_labels is not None, f"{
            self.__class__.__name__
        } requires train_features and train_labels"

        # Convert to numpy if needed
        train_features = (
            train_features.numpy()
            if isinstance(train_features, torch.Tensor)
            else train_features
        )
        train_labels = (
            train_labels.numpy()
            if isinstance(train_labels, torch.Tensor)
            else train_labels
        )

        # Compute MI once and cache ranking
        if self._cached_ranking is None:
            mi_scores = mutual_info_classif(
                train_features, train_labels, random_state=self.seed
            )
            # Sort ascending by MI score (worst first)
            self._cached_ranking = np.argsort(mi_scores)

        return self._cached_ranking[:n_features_select].tolist()


class ZeroInitializer(AFAInitializer):
    """
    Cold-start initializer that selects no features.
    """

    def select_features(
        self,
        n_features_total: int,
        n_features_select: int,
        train_features: Features | None = None,
        train_labels: Label | None = None,
    ) -> list[int]:
        return []


@final
class AACODefaultInitializer(AFAInitializer):
    """
    AACO paper's deterministic initial feature selection.

    Uses dataset-specific initial features as per the original paper:
    - cube: feature 6
    - mnist/fashionmnist: feature 100
    - others: middle feature
    """

    DATASET_INITIAL_FEATURES = {
        "cube": 6,
        "cubesimple": 3,
        "grid": 1,
        "gas10": 6,
        "mnist": 100,
        "fashionmnist": 100,
        "afacontext": 0,  # context feature first
    }

    def __init__(self, config: AACODefaultInitializerConfig):
        super().__init__()
        self.dataset_name = config.dataset_name.lower()

    def select_features(
        self,
        n_features_total: int,
        n_features_select: int,
        train_features: Features | None = None,
        train_labels: Label | None = None,
    ) -> list[int]:
        # AACO always starts with exactly 1 feature
        assert n_features_select == 1, (
            "AACO default initializer selects exactly 1 feature"
        )

        initial = self.DATASET_INITIAL_FEATURES.get(
            self.dataset_name,
            n_features_total // 2,  # fallback: middle feature
        )
        return [initial]
