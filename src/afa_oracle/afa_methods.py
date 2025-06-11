import torch
import numpy as np

from torch import Tensor
from pathlib import Path
from typing import Dict, List

from common.afa_methods import AFAMethod
from common.models import MaskedMLPClassifier
from .aco_core import ACOOracle


class ACOOracleMethod(AFAMethod):
    """
    Wraps the non-greedy ACO oracle for the benchmark.
    """

    def __init__(self,
                 predictor: MaskedMLPClassifier,
                 X_train: torch.Tensor,
                 y_train: torch.Tensor,
                 costs: torch.Tensor,
                 k: int = 5,
                 alpha: float = 0.01,
                 method: str = "full",  # "greedy" or "full"
                 device: torch.device = torch.device("cpu")):
        assert method in ["greedy", "full"], f"Unknown method: {method}"

        self._device = device
        self.predictor = predictor.to(self._device).eval()
        self.costs = costs.to(self._device)
        self.alpha = alpha
        self.method = method
        self.k = k

        # Store for saving/loading
        self.X_train_tensor = X_train.to(self._device)
        self.y_train_tensor = y_train.to(self._device)

        # Convert one-hot labels to class indices
        if y_train.dim() > 1 and y_train.shape[1] > 1:
            # One-hot encoded labels
            y_train_indices = y_train.argmax(dim=-1)
        else:
            # Already class indices
            y_train_indices = y_train.long()

        self.core = ACOOracle(
            X_train.cpu().numpy(),
            y_train_indices.cpu().numpy(),
            predictor=self.predictor,
            k_neighbors=k,
            feature_costs=costs.cpu().numpy(),
        )

    @classmethod
    def load(cls, path: Path, device: torch.device = torch.device("cpu")) -> "ACOOracleMethod":
        """
        Load saved ACO method.
        """
        device_obj = torch.device(device)
        state = torch.load(path / "state.pt", map_location=device_obj)

        predictor = MaskedMLPClassifier(**state["predictor_kwargs"])
        predictor.load_state_dict(torch.load(
            path / "predictor.pt", map_location=device_obj))

        return cls(
            predictor=predictor,
            X_train=state["X_train"],
            y_train=state["y_train"],
            costs=state["costs"],
            k=state["k"],
            alpha=state["alpha"],
            method=state["method"],
            device=device
        )

    def save(self, path: Path) -> None:
        """
        Save ACO method.
        """
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.predictor.state_dict(), path / "predictor.pt")
        torch.save(
            {
                "predictor_kwargs": {
                    "n_features": self.predictor.n_features,
                    "n_classes": self.predictor.n_classes,
                    "num_cells": self.predictor.num_cells,
                    "dropout": self.predictor.dropout,
                },
                "X_train": self.X_train_tensor.cpu(),
                "y_train": self.y_train_tensor.cpu(),
                "costs": self.costs.cpu(),
                "k": self.k,
                "alpha": self.alpha,
                "method": self.method,
            },
            path / "state.pt",
        )

    def reset(self, batch: Dict[str, Tensor]) -> None:
        """
        Reset for new batch of instances.
        """
        self.x_true = batch["x"].to(self._device)          # (B,d)
        # (B,) - only for final evaluation
        self.y_true = batch["y"].to(self._device)
        self.mask = torch.zeros_like(self.x_true).bool()  # (B,d)
        self.observed = torch.zeros_like(self.x_true)     # (B,d)

        # Track acquisition for logging
        self.acquisition_history = []

    def select_feature(self) -> Tensor:
        """
        Select next feature(s) to acquire for each instance in batch.
        """
        B, d = self.x_true.shape
        actions: List[int] = []

        for b in range(B):
            # Get currently observed features
            obs_idx = self.mask[b].nonzero(as_tuple=False).flatten().tolist()
            remaining = set(range(d)) - set(obs_idx)

            if not remaining:
                actions.append(-1)  # Stop acquisition
                continue

            # Get observed values as numpy array
            x_obs_numpy = self.observed[b].cpu().numpy()

            # Select feature using ACO
            if self.method == "greedy":
                feat = self.core.greedy_select_feature(
                    x_obs_numpy, set(obs_idx), remaining, self.alpha
                )
            else:  # "full"
                feat = self.core.full_aco_select_feature(
                    x_obs_numpy, set(obs_idx), remaining, self.alpha
                )

            if feat is None:
                actions.append(-1)  # Stop acquisition
            else:
                # Acquire the feature
                self.mask[b, feat] = True
                self.observed[b, feat] = self.x_true[b, feat]
                actions.append(feat)

        # Track for logging
        self.acquisition_history.append(actions.copy())

        return torch.tensor(actions, device=self._device, dtype=torch.long)

    def get_predictions(self) -> Tensor:
        """Make predictions with currently observed features."""
        logits = self.predictor(self.observed, self.mask.float())
        return torch.softmax(logits, -1)

    @property
    def device(self) -> torch.device:
        """
        Return the current device the method is on.
        """
        return self._device

    def to(self, device: torch.device) -> "ACOOracleMethod":
        """
        Move the object to the specified device.
        """
        new_device = torch.device(device)
        if new_device == self._device:
            return self

        # Move all tensors to new device
        self._device = new_device
        self.predictor = self.predictor.to(new_device)
        self.costs = self.costs.to(new_device)
        self.X_train_tensor = self.X_train_tensor.to(new_device)
        self.y_train_tensor = self.y_train_tensor.to(new_device)

        # Update core oracle's device
        self.core.device = new_device
        self.core.predictor = self.core.predictor.to(new_device)

        # Move current state if it exists
        if hasattr(self, 'x_true'):
            self.x_true = self.x_true.to(new_device)
            self.y_true = self.y_true.to(new_device)
            self.mask = self.mask.to(new_device)
            self.observed = self.observed.to(new_device)

        return self

    def get_acquisition_stats(self) -> Dict:
        """
        Get statistics about feature acquisition.
        """
        if not self.acquisition_history:
            return {}

        # Count total features acquired per instance
        total_acquired = []
        for b in range(len(self.acquisition_history[0])):
            count = 0
            for step in self.acquisition_history:
                if step[b] != -1:
                    count += 1
            total_acquired.append(count)

        return {
            "avg_features_acquired": np.mean(total_acquired),
            "min_features_acquired": np.min(total_acquired),
            "max_features_acquired": np.max(total_acquired),
            "total_steps": len(self.acquisition_history)
        }

    def select(self, masked_features: Tensor, feature_mask: Tensor) -> Tensor:
        """
        Select next feature to acquire (required by AFAMethod protocol).

        Args:
            masked_features: (B, d) currently observed features
            feature_mask: (B, d) binary mask of observed features

        Returns:
            AFASelection: (B, 1) next feature indices (1-based) or 0 to stop
        """
        B, d = masked_features.shape
        actions = []

        for b in range(B):
            # Get currently observed features for this instance
            mask_b = feature_mask[b].bool()
            obs_idx = mask_b.nonzero(as_tuple=False).flatten().tolist()
            remaining = set(range(d)) - set(obs_idx)

            if not remaining:
                actions.append(0)  # No more features to acquire (0 means stop)
                continue

            # Get observed values as numpy array
            x_obs_numpy = masked_features[b].cpu().numpy()

            # Select feature using ACO
            if self.method == "greedy":
                feat = self.core.greedy_select_feature(
                    x_obs_numpy, set(obs_idx), remaining, self.alpha
                )
            else:  # "full"
                feat = self.core.full_aco_select_feature(
                    x_obs_numpy, set(obs_idx), remaining, self.alpha
                )

            if feat is None:
                actions.append(0)  # Stop acquisition (0 means stop)
            else:
                actions.append(feat + 1)  # Convert to 1-based indexing

        return torch.tensor(actions, device=masked_features.device, dtype=torch.long).unsqueeze(1)

    def select_batch(self, masked_features: Tensor, feature_mask: Tensor, budget: int) -> tuple[list, Tensor]:
        """
        Select features for a batch of instances (compatible with eval_afa_method).

        Args:
            masked_features: (B, d) currently observed features
            feature_mask: (B, d) binary mask of observed features
            budget: maximum number of features to acquire

        Returns:
            selected_features: list of lists of selected feature indices
            final_mask: (B, d) final feature mask
        """
        B, d = masked_features.shape

        # Convert to the format expected by reset
        batch = {
            "x": masked_features.clone(),  # Full features (we'll simulate acquiring them)
            # Dummy labels for selection
            "y": torch.zeros(B, self.core.predictor.n_classes)
        }

        # Reset with current state
        self.reset(batch)

        # Override current observation state
        self.observed = masked_features.clone()
        self.mask = feature_mask.bool()

        selected_features = [[] for _ in range(B)]

        # Sequential feature acquisition
        for step in range(budget):
            actions = self.select_feature()

            # Track selections and stop when no more features to acquire
            all_stopped = True
            for b, action in enumerate(actions):
                if action.item() != -1:
                    selected_features[b].append(action.item())
                    all_stopped = False

            if all_stopped:
                break

        return selected_features, self.mask.float()

    def predict(self, masked_features: Tensor, feature_mask: Tensor) -> Tensor:
        """
        Make predictions given masked features (compatible with eval_afa_method).

        Args:
            masked_features: (B, d) observed features
            feature_mask: (B, d) binary mask of observed features

        Returns:
            predictions: (B, n_classes) class probabilities
        """
        # Temporarily set state for prediction
        old_observed = getattr(self, 'observed', None)
        old_mask = getattr(self, 'mask', None)

        self.observed = masked_features.to(self._device)
        self.mask = feature_mask.bool().to(self._device)

        # Get predictions
        predictions = self.get_predictions()

        # Restore old state
        if old_observed is not None:
            self.observed = old_observed
        if old_mask is not None:
            self.mask = old_mask

        return predictions
