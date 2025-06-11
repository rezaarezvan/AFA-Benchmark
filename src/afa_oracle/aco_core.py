import torch
import numpy as np
import torch.nn as nn

from typing import Set, Optional
from sklearn.neighbors import NearestNeighbors


class ACOOracle:
    """
    Acquisition Conditioned Oracle for active feature acquisition.
    https://arxiv.org/pdf/2302.13960
    """

    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, predictor,
                 k_neighbors: int = 5, feature_costs: Optional[np.ndarray] = None):
        assert X_train.shape[0] == len(
            y_train), f"X_train and y_train size mismatch"
        assert k_neighbors > 0, f"k_neighbors must be positive, got {
            k_neighbors}"

        self.X_train = X_train
        self.y_train = y_train
        self.predictor = predictor
        self.k = k_neighbors
        self.n_features = X_train.shape[1]
        self.device = next(predictor.parameters()).device

        # Feature costs (default to uniform cost of 1.0)
        if feature_costs is None:
            self.feature_costs = np.ones(self.n_features)
        else:
            assert len(feature_costs) == self.n_features, f"Expected {
                self.n_features} costs, got {len(feature_costs)}"
            self.feature_costs = np.array(feature_costs)

        # Build k-NN index ONCE for the full feature space
        self.knn_index = NearestNeighbors(
            n_neighbors=k_neighbors, metric="euclidean")
        self.knn_index.fit(X_train)

    def find_neighbors(self, x_observed: np.ndarray, observed_indices: Set[int]) -> tuple:
        """
        Find k nearest neighbors based on observed features only.

        Args:
            x_observed (np.ndarray): The observed feature vector.
            observed_indices (Set[int]): Indices of features that have been observed.
        Returns:
            tuple: Indices of the k nearest neighbors and their distances.
        """
        if len(observed_indices) == 0:
            # If no features observed, return random neighbors
            n_samples = len(self.X_train)
            neighbor_indices = np.random.choice(
                n_samples, size=min(self.k, n_samples), replace=False)
            distances = np.ones(len(neighbor_indices))
            return neighbor_indices, distances

        # Create partial feature vectors for similarity computation
        observed_list = sorted(list(observed_indices))
        x_partial = x_observed[observed_list].reshape(1, -1)
        X_train_partial = self.X_train[:, observed_list]

        # Create new knn index for the partial space
        knn_partial = NearestNeighbors(n_neighbors=self.k, metric="euclidean")
        knn_partial.fit(X_train_partial)
        distances, neighbor_indices = knn_partial.kneighbors(x_partial)

        return neighbor_indices[0], distances[0]

    def estimate_expected_loss(self, x_observed: np.ndarray, observed_indices: Set[int],
                               candidate_feature: int, alpha: float = 0.0) -> float:
        """
        Estimate expected loss if we acquire the candidate feature.

        Args:
            x_observed (np.ndarray): The observed feature vector.
            observed_indices (Set[int]): Indices of features that have been observed.
            candidate_feature (int): The feature index to acquire.
            alpha (float): Cost coefficient for feature acquisition.

        Returns:
            float: Estimated expected loss after acquiring the candidate feature.
        """
        neighbor_indices, _ = self.find_neighbors(x_observed, observed_indices)

        new_observed_indices = observed_indices | {candidate_feature}
        total_loss = 0.0

        for neighbor_idx in neighbor_indices:
            x_new = self.X_train[neighbor_idx].copy()

            # Overwrite with actual observed values
            for idx in observed_indices:
                x_new[idx] = x_observed[idx]
            # Candidate value comes from neighbor (already there)

            # Create mask for new observed indices
            mask = np.zeros(self.n_features)
            for idx in new_observed_indices:
                mask[idx] = 1.0

            # Compute prediction and loss
            with torch.no_grad():
                x_tensor = torch.tensor(
                    x_new, dtype=torch.float32).unsqueeze(0).to(self.device)
                mask_tensor = torch.tensor(
                    mask, dtype=torch.float32).unsqueeze(0).to(self.device)

                y_neighbor = torch.tensor(
                    [self.y_train[neighbor_idx]], dtype=torch.long).to(self.device)

                logits = self.predictor(x_tensor, mask_tensor)
                loss = nn.CrossEntropyLoss()(logits, y_neighbor)
                total_loss += loss.item()

        expected_loss = total_loss / len(neighbor_indices)
        cost_term = alpha * self.feature_costs[candidate_feature]

        return expected_loss + cost_term

    def greedy_select_feature(self, x_observed: np.ndarray, observed_indices: Set[int],
                              remaining_candidates: Set[int], alpha: float = 0.0) -> Optional[int]:
        """
        Greedily select the next best feature to acquire.

        Args:
            x_observed (np.ndarray): The observed feature vector.
            observed_indices (Set[int]): Indices of features that have been observed.
            remaining_candidates (Set[int]): Indices of features that can still be acquired.
            alpha (float): Cost coefficient for feature acquisition.
        Returns:
            Optional[int]: The index of the feature to acquire next, or None if no candidates left.
        """
        if len(remaining_candidates) == 0:
            return None

        best_feature = None
        best_loss = float("inf")

        for candidate in remaining_candidates:
            expected_loss = self.estimate_expected_loss(
                x_observed, observed_indices, candidate, alpha)

            if expected_loss < best_loss:
                best_loss = expected_loss
                best_feature = candidate

        return best_feature

    def estimate_subset_loss(self, x_observed: np.ndarray, observed_indices: Set[int],
                             candidate_subset: Set[int], alpha: float = 0.0) -> float:
        """
        Estimate expected loss if we acquire all features in candidate_subset.

        Args:
            x_observed (np.ndarray): The observed feature vector.
            observed_indices (Set[int]): Indices of features that have been observed.
            candidate_subset (Set[int]): Indices of features to acquire.
            alpha (float): Cost coefficient for feature acquisition.
        Returns:
            float: Estimated expected loss after acquiring the candidate subset.
        """
        neighbor_indices, _ = self.find_neighbors(x_observed, observed_indices)

        new_observed_indices = observed_indices | candidate_subset
        total_loss = 0.0

        for neighbor_idx in neighbor_indices:
            x_new = self.X_train[neighbor_idx].copy()

            # Overwrite with actual observed values
            for idx in observed_indices:
                x_new[idx] = x_observed[idx]

            # Create mask
            mask = np.zeros(self.n_features)
            for idx in new_observed_indices:
                mask[idx] = 1.0

            # Compute prediction and loss
            with torch.no_grad():
                x_tensor = torch.tensor(
                    x_new, dtype=torch.float32).unsqueeze(0).to(self.device)
                mask_tensor = torch.tensor(
                    mask, dtype=torch.float32).unsqueeze(0).to(self.device)

                y_neighbor = torch.tensor(
                    [self.y_train[neighbor_idx]], dtype=torch.long).to(self.device)

                logits = self.predictor(x_tensor, mask_tensor)
                loss = nn.CrossEntropyLoss()(logits, y_neighbor)
                total_loss += loss.item()

        expected_loss = total_loss / len(neighbor_indices)
        cost_term = alpha * sum(self.feature_costs[list(candidate_subset)])

        return expected_loss + cost_term

    def full_aco_select_feature(self, x_observed: np.ndarray, observed_indices: Set[int],
                                remaining_candidates: Set[int], alpha: float = 0.0,
                                n_subsets: int = 1000) -> Optional[int]:
        """
        Full ACO: consider joint acquisition of multiple features, then select one.

        Args:
            x_observed (np.ndarray): The observed feature vector.
            observed_indices (Set[int]): Indices of features that have been observed.
            remaining_candidates (Set[int]): Indices of features that can still be acquired.
            alpha (float): Cost coefficient for feature acquisition.
            n_subsets (int): Number of random subsets to consider.

        Returns:
            Optional[int]: The index of the feature to acquire next, or None if no candidates left.
        """
        if len(remaining_candidates) == 0:
            return None

        best_subset = None
        best_loss = float("inf")

        # Consider subsets of remaining candidates
        candidates_list = list(remaining_candidates)

        for _ in range(n_subsets):
            # Sample random subset size (1 to min(5, len(candidates)))
            max_subset_size = min(5, len(candidates_list))
            subset_size = np.random.randint(1, max_subset_size + 1)

            # Sample random subset
            subset = set(np.random.choice(candidates_list,
                         size=subset_size, replace=False))

            # Estimate expected loss for this subset
            expected_loss = self.estimate_subset_loss(
                x_observed, observed_indices, subset, alpha)

            if expected_loss < best_loss:
                best_loss = expected_loss
                best_subset = subset

        if best_subset and len(best_subset) > 1:
            # Break ties by single feature giving biggest marginal loss drop
            best_single_feature = None
            best_single_loss = float("inf")

            for feature in best_subset:
                single_loss = self.estimate_expected_loss(
                    x_observed, observed_indices, feature, alpha)
                if single_loss < best_single_loss:
                    best_single_loss = single_loss
                    best_single_feature = feature

            return best_single_feature
        elif best_subset:
            return list(best_subset)[0]
        else:
            return np.random.choice(candidates_list)
