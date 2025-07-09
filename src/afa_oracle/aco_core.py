import torch
import logging
import numpy as np
import torch.nn as nn

from itertools import chain, combinations
from common.custom_types import AFAClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Tuple, Callable

logger = logging.getLogger(__name__)


def powerset(iterable, max_size: Optional[int] = None):
    """
    Generate all possible subsets (powerset) of the iterable.
    """
    s = list(iterable)
    max_r = len(s) + 1 if max_size is None else min(max_size + 1, len(s) + 1)
    return [
        list(x)
        for x in chain.from_iterable(combinations(s, r) for r in range(0, max_r))
    ]


class KNNDensityEstimator:
    """
    k-Nearest Neighbors density estimator for ACO.
    """

    def __init__(self, k: int = 5, metric: str = 'euclidean', standardize: bool = True):
        self.k = k
        self.metric = metric
        self.standardize = standardize
        self.scaler = StandardScaler() if standardize else None
        self.knn = NearestNeighbors(n_neighbors=k, metric=metric)
        self.X_train = None
        self.y_train = None

    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor):
        """
        Fit the k-NN estimator on training data.
        """

        self.X_train = X_train.clone().detach()
        self.y_train = y_train.clone().detach()

        # Convert to numpy for sklearn
        X_np = X_train.cpu().numpy()

        if self.standardize and self.scaler is not None:
            X_np = self.scaler.fit_transform(X_np)

        self.knn.fit(X_np)

    def estimate_conditional_expectation(
        self,
        x_observed: torch.Tensor,
        observed_mask: torch.Tensor,
        target_mask: torch.Tensor,
        loss_fn: Callable,
        predictor: AFAClassifier
    ) -> float:
        """
        Estimate E[loss(predictor(x_observed, x_target), y) | x_observed]
        using k-nearest neighbors.
        """

        if self.X_train is None:
            raise ValueError(
                "Must call fit() before estimate_conditional_expectation()")

        # Get observed features only
        observed_indices = observed_mask.nonzero(as_tuple=True)[0]

        # Check if no features are observed yet (handled by select_next_feature now)
        if len(observed_indices) == 0:
            # This shouldn't happen now with the deterministic initial selection
            return float('inf')

        x_obs_values = x_observed[observed_indices]

        # Standardize query point if needed
        x_obs_np = x_obs_values.unsqueeze(0).cpu().numpy()
        if self.standardize and self.scaler is not None:
            # Only transform the observed features
            full_x = torch.zeros(
                self.X_train.shape[1], device=x_obs_values.device)
            full_x[observed_indices] = x_obs_values
            full_x_np = full_x.unsqueeze(0).cpu().numpy()
            full_x_scaled = self.scaler.transform(full_x_np)[0]
            x_obs_scaled = full_x_scaled[observed_indices.cpu().numpy()]
            x_obs_np = x_obs_scaled.reshape(1, -1)

        # Find k nearest neighbors based on observed features only
        X_train_observed = self.X_train[:,
                                        observed_indices.to(self.X_train.device)]
        if self.standardize and self.scaler is not None:
            X_train_full_scaled = self.scaler.transform(
                self.X_train.cpu().numpy())
            X_train_observed = torch.tensor(X_train_full_scaled)[
                :, observed_indices.to(self.X_train.device)]

        # Fit temporary k-NN on observed features only
        temp_knn = NearestNeighbors(n_neighbors=self.k, metric=self.metric)
        temp_knn.fit(X_train_observed.cpu().numpy())

        # Get neighbors
        _, neighbor_indices = temp_knn.kneighbors(x_obs_np)
        neighbor_indices = neighbor_indices[0]

        # Compute loss for each neighbor
        losses = []
        for idx in neighbor_indices:
            # Get neighbor's full feature vector and label
            neighbor_x = self.X_train[idx]
            neighbor_y = self.y_train[idx]

            # Create masked features using target mask
            masked_neighbor_x = neighbor_x.clone()
            # Hide unobserved features
            masked_neighbor_x[~target_mask.to(neighbor_x.device)] = 0

            # Get prediction
            pred = predictor(
                masked_neighbor_x.unsqueeze(0),
                target_mask.to(neighbor_x.device).unsqueeze(0)
            )

            # Compute loss
            loss = loss_fn(pred, neighbor_y.unsqueeze(0))
            losses.append(loss.item())

        return float(np.mean(losses))


class SubsetSearchStrategy:
    """
    Strategy for searching over feature subsets.
    """

    def __init__(
        self,
        exhaustive_threshold: int = 12,
        max_samples: int = 10000,
        max_subset_size: Optional[int] = None
    ):
        self.exhaustive_threshold = exhaustive_threshold
        self.max_samples = max_samples
        self.max_subset_size = max_subset_size

    def generate_subsets(self, available_features: List[int]) -> List[List[int]]:
        """Generate subsets to search over."""
        n_features = len(available_features)

        if n_features <= self.exhaustive_threshold:
            # Use exhaustive search for small feature spaces
            return powerset(available_features, self.max_subset_size)
        else:
            # Use random sampling for large feature spaces
            return self._random_subsets(available_features)

    def _random_subsets(self, available_features: List[int]) -> List[List[int]]:
        """Generate random subsets."""
        subsets = []
        n_features = len(available_features)
        max_size = min(self.max_subset_size or n_features, n_features)

        # Always include empty set
        subsets.append([])

        # Generate random subsets (bias towards smaller sizes)
        # larger weight for smaller i+1
        size_weights = [max_size - i for i in range(max_size)]
        for _ in range(self.max_samples - 1):
            # sample size  in [1, max_size] with bias
            size = np.random.choice(
                np.arange(1, max_size + 1), p=np.array(size_weights) / sum(size_weights)
            )
            subset = list(np.random.choice(
                available_features, size, replace=False))
            if subset not in subsets:
                subsets.append(subset)

        return subsets


class ACOOracle:
    """
    Acquisition Conditioned Oracle implementation.
    """

    def __init__(
        self,
        density_estimator: KNNDensityEstimator,
        subset_search: SubsetSearchStrategy,
        acquisition_cost: float = 0.05,
        hide_val: float = 10.0
    ):
        self.density_estimator = density_estimator
        self.subset_search = subset_search
        self.acquisition_cost = acquisition_cost
        self.hide_val = hide_val
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor):
        """
        Fit the oracle on training data.
        """

        self.density_estimator.fit(X_train, y_train)

    def find_optimal_subset(
        self,
        x_observed: torch.Tensor,
        observed_mask: torch.Tensor,
        predictor: AFAClassifier
    ) -> Tuple[List[int], float]:
        """
        Find optimal subset of features to acquire according to ACO objective.
        """

        # Get available features (not yet observed)
        all_features = set(range(len(x_observed)))
        observed_features = set(
            observed_mask.nonzero(as_tuple=True)[0].tolist())
        available_features = list(all_features - observed_features)

        if not available_features:
            return [], float('inf')

        # Generate candidate subsets
        candidate_subsets = self.subset_search.generate_subsets(
            available_features)

        logger.info(f"Evaluating {len(candidate_subsets)} candidate subsets")

        best_subset = []
        best_loss = float('inf')

        for i, subset in enumerate(candidate_subsets):
            if i % 100 == 0:  # Log every 100 subsets
                logger.info(f"Processed {i}/{len(candidate_subsets)} subsets")

            # Create target mask (observed + subset)
            target_mask = observed_mask.clone()
            if subset:  # Only update if subset is not empty
                subset_tensor = torch.tensor(subset, dtype=torch.long)
                target_mask[subset_tensor] = True

            # Estimate expected loss for this subset
            expected_loss = self.density_estimator.estimate_conditional_expectation(
                x_observed, observed_mask, target_mask, self.loss_fn, predictor
            )

            # Add acquisition cost
            total_cost = expected_loss + self.acquisition_cost * len(subset)

            if total_cost < best_loss:
                best_loss = total_cost
                best_subset = subset

        logger.info(f"Finished evaluating subsets. Best subset: {best_subset}")
        return best_subset, best_loss

    def select_next_feature(
        self,
        x_observed: torch.Tensor,
        observed_mask: torch.Tensor,
        predictor: AFAClassifier
    ) -> Optional[int]:
        """
        Select next feature to acquire using ACO policy.
        """

        # Check if no features are observed yet (initial case)
        observed_indices = observed_mask.nonzero(as_tuple=True)[0]
        if len(observed_indices) == 0:
            # Deterministic initial feature selection (like original AACO)
            # Use feature 6 for cube, 100 for MNIST, etc.
            n_features = len(x_observed)
            if n_features == 20:  # cube dataset
                return 6
            elif n_features == 784:  # MNIST dataset
                return 100
            else:
                # Default: select middle feature
                return n_features // 2

        # Regular ACO logic for subsequent features
        optimal_subset, best_loss = self.find_optimal_subset(
            x_observed, observed_mask, predictor
        )

        if not optimal_subset:
            return None  # No beneficial acquisition

        # If multiple features in optimal subset, select the one with lowest individual loss
        if len(optimal_subset) == 1:
            return optimal_subset[0]
        else:
            # Evaluate each feature individually and pick the best
            best_feature = None
            best_individual_loss = float('inf')

            for feature in optimal_subset:
                individual_mask = observed_mask.clone()
                individual_mask[feature] = True

                individual_loss = self.density_estimator.estimate_conditional_expectation(
                    x_observed, observed_mask, individual_mask, self.loss_fn, predictor
                )

                if individual_loss < best_individual_loss:
                    best_individual_loss = individual_loss
                    best_feature = feature

            return best_feature


def create_aco_oracle(
    k_neighbors: int = 5,
    acquisition_cost: float = 0.05,
    exhaustive_threshold: int = 12,
    subset_search_size: int = 10000,
    max_subset_size: Optional[int] = None,
    hide_val: float = 10.0,
    distance_metric: str = 'euclidean',
    standardize_features: bool = True
) -> ACOOracle:
    """
    Factory function to create ACO oracle with specified parameters.
    """

    density_estimator = KNNDensityEstimator(
        k=k_neighbors,
        metric=distance_metric,
        standardize=standardize_features
    )

    subset_search = SubsetSearchStrategy(
        exhaustive_threshold=exhaustive_threshold,
        max_samples=subset_search_size,
        max_subset_size=max_subset_size
    )

    return ACOOracle(
        density_estimator=density_estimator,
        subset_search=subset_search,
        acquisition_cost=acquisition_cost,
        hide_val=hide_val
    )
