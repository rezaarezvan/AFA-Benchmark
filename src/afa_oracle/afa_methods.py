import torch
import logging

from pathlib import Path
from dataclasses import dataclass
from afa_oracle.aaco_core import AACOOracle
from typing import Self, final, override
from common.custom_types import (
    AFAMethod, AFAClassifier, AFASelection, FeatureMask,
    Label, MaskedFeatures, AFADataset
)

logger = logging.getLogger(__name__)


@dataclass
@final
class AACOAFAMethod(AFAMethod):
    aaco_oracle: AACOOracle
    dataset_name: str
    _device: torch.device = torch.device("cpu")

    def __post_init__(self):
        # Move oracle to device if needed
        pass

    @override
    def select(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> AFASelection:
        """
        Select next feature using their AACO implementation
        """
        original_device = masked_features.device

        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)

        # AACO works on single instances, process batch one by one
        batch_size = masked_features.shape[0]
        selections = []

        for i in range(batch_size):
            x_obs = masked_features[i]
            obs_mask = feature_mask[i]

            # Get next feature from AACO oracle
            next_feature = self.aaco_oracle.select_next_feature(
                x_obs, obs_mask, instance_idx=i
            )

            if next_feature is not None:
                selections.append(next_feature)
            else:
                selections.append(-1)  # Terminate signal

        selection_tensor = torch.tensor(
            selections, dtype=torch.long).unsqueeze(-1)
        return selection_tensor.to(original_device)

    @override
    def predict(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Label:
        """
        Make prediction using classifier approach
        """
        original_device = masked_features.device

        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)

        batch_size = masked_features.shape[0]
        n_classes = self.aaco_oracle.y_train.shape[-1] if len(
            self.aaco_oracle.y_train.shape) > 1 else self.aaco_oracle.y_train.max().item() + 1

        predictions = torch.zeros(batch_size, n_classes)

        for i in range(batch_size):
            x_obs = masked_features[i]
            obs_mask = feature_mask[i]

            x_masked = x_obs * obs_mask.float() - (1 - obs_mask.float()) * \
                self.aaco_oracle.hide_val
            x_with_mask = torch.cat([x_masked, obs_mask.float()])

            pred = self.aaco_oracle.classifier(
                x_with_mask.unsqueeze(0), torch.tensor([0]))
            pred = pred.squeeze()

            # Handle different types of classifier outputs
            # Output is logits or unnormalized, apply softmax
            if torch.any(pred < 0) or torch.any(pred > 1) or not torch.allclose(pred.sum(), torch.tensor(1.0), atol=1e-6):
                pred = torch.softmax(pred, dim=0)

            # Final safety check: ensure valid probabilities
            pred = torch.clamp(pred, min=1e-8, max=1.0)  # Clamp to valid range
            pred = pred / pred.sum()  # Renormalize to sum to 1

            predictions[i] = pred

        return predictions.to(original_device)

    @override
    def save(self, path: Path):
        """
        Save the AACO method
        """
        path.mkdir(parents=True, exist_ok=True)

        # Save oracle parameters
        oracle_state = {
            'k_neighbors': self.aaco_oracle.k_neighbors,
            'acquisition_cost': self.aaco_oracle.acquisition_cost,
            'hide_val': self.aaco_oracle.hide_val,
            'dataset_name': self.dataset_name,
        }

        torch.save(oracle_state, path / 'aaco_oracle.pt')
        logger.info(f"Saved AACO method to {path}")

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        """
        Load the AACO method
        """
        oracle_state = torch.load(path / 'aaco_oracle.pt', map_location=device)

        # Recreate oracle
        aaco_oracle = AACOOracle(
            k_neighbors=oracle_state['k_neighbors'],
            acquisition_cost=oracle_state['acquisition_cost'],
            hide_val=oracle_state['hide_val'],
            dataset_name=oracle_state['dataset_name']
        )

        method = cls(
            aaco_oracle=aaco_oracle,
            dataset_name=oracle_state['dataset_name'],
            _device=device
        )

        logger.info(f"Loaded AACO method from {path}")
        return method

    @override
    def to(self, device: torch.device) -> Self:
        """
        Move method to device
        """
        self._device = device
        # Move oracle data to device if fitted
        if self.aaco_oracle.X_train is not None:
            self.aaco_oracle.X_train = self.aaco_oracle.X_train.to(device)
            self.aaco_oracle.y_train = self.aaco_oracle.y_train.to(device)
        return self

    @property
    @override
    def device(self) -> torch.device:
        """
        Get device
        """
        return self._device


def create_aaco_method(
    k_neighbors: int = 5,
    acquisition_cost: float = 0.05,
    hide_val: float = 10.0,
    dataset_name: str = "cube",
    device: torch.device = torch.device("cpu")
) -> AACOAFAMethod:
    """
    Factory function to create AACO method
    """
    aaco_oracle = AACOOracle(
        k_neighbors=k_neighbors,
        acquisition_cost=acquisition_cost,
        hide_val=hide_val,
        dataset_name=dataset_name
    )

    return AACOAFAMethod(
        aaco_oracle=aaco_oracle,
        dataset_name=dataset_name,
        _device=device
    )
