from abc import ABC, abstractmethod

import numpy as np

from afabench.common.custom_types import Features, Label


class WarmStartStrategy(ABC):
    """
    Abstract base class for selecting initial features in warm-start scenarios.

    Subclasses must implement select_features() to define feature selection logic.
    """

    def __init__(self, seed: int | None = None):
        """
        Initialize the warm-start strategy.

        Args:
            seed: Random seed for reproducibility. If None, results may vary.
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    @abstractmethod
    def select_features(
        self,
        n_features_total: int,
        n_features_select: int,
        train_features: Features | None = None,
        train_labels: Label | None = None,
    ) -> list[int]:
        """
        Select which features to provide as warm-start.

        Args:
            n_features_total: Total number of features available
            n_features_select: Number of features to select for warm-start
            train_features: Training features (shape: [n_samples, n_features])
                           Required for data-dependent strategies
            train_labels: Training labels (shape: [n_samples,])
                         Required for data-dependent strategies

        Returns:
            List of feature indices to warm-start with (length = n_features_select)

        Raises:
            ValueError: If required data is not provided for data-dependent strategies
        """

    def __repr__(self) -> str:
        """String representation for logging."""
        return f"{self.__class__.__name__}(seed={self.seed})"
