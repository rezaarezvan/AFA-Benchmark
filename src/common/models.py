from typing import Tuple

import lightning as pl
import torch
import torch.nn.functional as F
from jaxtyping import Float, Shaped
from torch import Tensor, nn, optim
from torchrl.modules import MLP

from afa_rl.custom_types import (
    Features,
    Label,
    MaskedClassifier,
    Logits,
    NNMaskedClassifier,
)
from afa_rl.shim2018.custom_types import Embedder, Embedding, EmbeddingClassifier
from afa_rl.utils import (
    get_feature_set,
    mask_data,
)
from common.custom_types import FeatureMask, MaskedFeatures


class LitMaskedClassifier(pl.LightningModule):
    """A lit module for classifiers that take masked features and feature masks as input."""

    def __init__(
        self,
        classifier: nn.Module,
        class_probabilities: Float[Tensor, "n_classes"],
        max_masking_probability=1.0,
        lr=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["classifier"])
        self.lr = lr
        self.classifier = classifier
        self.class_weight = 1 / class_probabilities
        self.class_weight = self.class_weight / torch.sum(self.class_weight)
        self.max_masking_probability = max_masking_probability

        # Initial masking probability
        self.masking_probability = 0.0

    def forward(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Logits:
        """Forward.

        Args:
            masked_features: currently observed features, with zeros for missing features
            feature_mask: indicator for missing features, 1 if feature is observed, 0 if missing

        Returns:
            logits

        """
        return self.classifier(masked_features, feature_mask)

    def on_train_epoch_start(self):
        # Masking probability uniformly distributed between 0 and self.max_masking_probability
        self.masking_probability = torch.rand(1).item() * self.max_masking_probability
        self.log("masking_probability", self.masking_probability, sync_dist=True)

    def training_step(self, batch, batch_idx):
        features: Features = batch[0]
        label: Label = batch[1]

        masked_features, feature_mask = mask_data(features, p=self.masking_probability)

        logits = self(masked_features, feature_mask)
        loss = F.cross_entropy(logits, label, weight=self.class_weight.to(logits.device))
        self.log("train_loss", loss)
        return loss

    def _get_loss_and_acc(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask, y: Tensor
    ) -> tuple[Tensor, Tensor]:
        logits = self(masked_features, feature_mask)
        loss = F.cross_entropy(logits, y, weight=self.class_weight.to(logits.device))
        predicted_class = torch.argmax(logits, dim=1)
        true_class = torch.argmax(y, dim=1)
        acc = (predicted_class == true_class).float().mean()
        return loss, acc

    def validation_step(self, batch, batch_idx):
        feature_values, y = batch

        feature_mask_half_observed = torch.randint(
            0, 2, feature_values.shape, device=feature_values.device
        )
        feature_values_half_observed = feature_values.clone()
        feature_values_half_observed[feature_mask_half_observed == 0] = 0
        loss, acc = self._get_loss_and_acc(
            feature_values_half_observed, feature_mask_half_observed, y
        )
        self.log("val_loss_half", loss)
        self.log("val_acc_half", acc)

        # WARNING: this block is only valid for AFAContextDataset
        feature_mask_optimal = torch.zeros_like(
            feature_values, dtype=torch.bool, device=feature_values.device
        )
        feature_mask_optimal[:, 0] = 1
        for i in range(feature_mask_optimal.shape[0]):
            context = feature_values[i, 0].int().item()
            feature_mask_optimal[i, 1 + context * 3 : 4 + context * 3] = 1
        feature_values_optimal = feature_values.clone()
        feature_values_optimal[feature_mask_optimal == 0] = 0
        loss, acc = self._get_loss_and_acc(
            feature_values_optimal, feature_mask_optimal, y
        )
        self.log("val_loss_optimal", loss)
        self.log("val_acc_optimal", acc)

        loss, acc = self._get_loss_and_acc(
            feature_values,
            torch.ones_like(
                feature_values, dtype=torch.bool, device=feature_values.device
            ),
            y,
        )
        self.log("val_loss_full", loss)
        self.log("val_acc_full", acc)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class MaskedMLPClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(n_features * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Logits:
        # Concatenate the masked features and the feature mask
        x = torch.cat((masked_features, feature_mask), dim=1)
        logits = self.classifier(x)
        return logits
