import torch
import numpy as np

from typing import Optional
from common.custom_types import Features, Label
from common.warm_start.base import WarmStartStrategy
from sklearn.feature_selection import mutual_info_classif


class FixedRandomStrategy(WarmStartStrategy):
    """
    Select random features once and reuse them across all episodes.
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self._cached_features: Optional[list[int]] = None

    def select_features(
        self,
        n_features_total: int,
        n_features_select: int,
        train_features: Optional[Features] = None,
        train_labels: Optional[Label] = None,
    ) -> list[int]:
        # Select once and cache
        if self._cached_features is None:
            self._cached_features = self.rng.choice(
                n_features_total, size=n_features_select, replace=False
            ).tolist()
        assert self._cached_features is not None
        return self._cached_features


class RandomPerEpisodeStrategy(WarmStartStrategy):
    """
    Select different random features for each episode.
    """

    def select_features(
        self,
        n_features_total: int,
        n_features_select: int,
        train_features: Optional[Features] = None,
        train_labels: Optional[Label] = None,
    ) -> list[int]:
        # Select new features each time
        return self.rng.choice(
            n_features_total, size=n_features_select, replace=False
        ).tolist()


class ManualStrategy(WarmStartStrategy):
    """
    Use explicitly specified feature indices.
    """

    def __init__(self, feature_indices: list[int], seed: Optional[int] = None):
        super().__init__(seed)
        self.feature_indices = feature_indices

    def select_features(
        self,
        n_features_total: int,
        n_features_select: int,
        train_features: Optional[Features] = None,
        train_labels: Optional[Label] = None,
    ) -> list[int]:
        assert len(self.feature_indices) == n_features_select, f"Expected {
            n_features_select
        } features, got {len(self.feature_indices)}"
        return self.feature_indices

    def __repr__(self) -> str:
        return f"ManualStrategy(features={self.feature_indices})"


class MutualInformationStrategy(WarmStartStrategy):
    """
    Select features with highest mutual information with target labels.
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self._cached_ranking: Optional[np.ndarray] = None

    def select_features(
        self,
        n_features_total: int,
        n_features_select: int,
        train_features: Optional[Features] = None,
        train_labels: Optional[Label] = None,
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


class LeastInformativeStrategy(WarmStartStrategy):
    """
    Select features with LOWEST mutual information (adversarial baseline).

    Useful for robustness testing - how well do methods perform with poor initialization?
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self._cached_ranking: Optional[np.ndarray] = None

    def select_features(
        self,
        n_features_total: int,
        n_features_select: int,
        train_features: Optional[Features] = None,
        train_labels: Optional[Label] = None,
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
