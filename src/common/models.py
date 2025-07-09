from typing import Tuple, final, override

import lightning as pl
import torch
import torch.nn.functional as F
from jaxtyping import Float, Shaped
from torch import Tensor, nn, optim
from torchrl.modules import MLP

from afa_rl.custom_types import (
    Features,
    Label,
    Logits,
)
from afa_rl.shim2018.custom_types import Embedder, Embedding, EmbeddingClassifier
from afa_rl.utils import (
    get_feature_set,
    mask_data,
)
from common.custom_types import AFAClassifier, FeatureMask, MaskedFeatures


@final
class LitMaskedMLPClassifier(pl.LightningModule):
    """A lit module for a MaskedMLPClassifier that takes masked features and feature masks as input."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        num_cells: tuple[int, ...] = (128, 128),
        dropout: float = 0.1,
        class_probabilities: Float[Tensor, "n_classes"] | None = None,
        min_masking_probability: float = 0.0,
        max_masking_probability: float = 1.0,
        lr: float = 1e-3,
    ):
        super().__init__()

        self.n_features = n_features
        self.n_classes = n_classes
        self.num_cells = num_cells
        self.dropout = dropout

        self.classifier = MaskedMLPClassifier(
            n_features=n_features,
            n_classes=n_classes,
            num_cells=num_cells,
            dropout=dropout,
        )

        if class_probabilities is None:
            self.class_probabilities = torch.ones(n_classes) / n_classes
        else:
            self.class_probabilities = class_probabilities
        self.class_weight = 1 / self.class_probabilities
        self.class_weight = self.class_weight / torch.sum(self.class_weight)

        self.lr = lr

        self.min_masking_probability = min_masking_probability
        self.max_masking_probability = max_masking_probability

        self.save_hyperparameters()

    @override
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

    @override
    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int):
        features: Features = batch[0]
        label: Label = batch[1]

        masking_probability = self.min_masking_probability + torch.rand(1).item() * (self.max_masking_probability-self.min_masking_probability)
        self.log("masking_probability", masking_probability, sync_dist=True)

        masked_features, feature_mask, _ = mask_data(features, p=masking_probability)

        logits = self(masked_features, feature_mask)
        loss = F.cross_entropy(
            logits, label, weight=self.class_weight.to(logits.device)
        )
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

    @override
    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int):
        feature_values, y = batch

        # Mask features with minimum probability -> see many features (observations)
        feature_mask_many_observations = torch.rand(
            feature_values.shape, device=feature_values.device
        ) > self.min_masking_probability
        feature_values_many_observations = feature_values.clone()
        feature_values_many_observations[feature_mask_many_observations == 0] = 0
        loss_many_observations, acc_many_observations = self._get_loss_and_acc(
            feature_values_many_observations, feature_mask_many_observations, y
        )
        self.log("val_loss_many_observations", loss_many_observations)
        self.log("val_acc_many_observations", acc_many_observations)

        # Mask features with maximum probability -> see few features (observations)
        feature_mask_few_observations = torch.rand(
            feature_values.shape, device=feature_values.device
        ) > self.max_masking_probability
        feature_values_few_observations = feature_values.clone()
        feature_values_few_observations[feature_mask_few_observations == 0] = 0
        loss_few_observations, acc_few_observations = self._get_loss_and_acc(
            feature_values_few_observations, feature_mask_few_observations, y
        )
        self.log("val_loss_few_observations", loss_few_observations)
        self.log("val_acc_few_observations", acc_few_observations)

    @override
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


@final
class MaskedMLPClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        num_cells: tuple[int, ...] = (128, 128),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.num_cells = num_cells
        self.dropout = dropout

        self.classifier = MLP(
            in_features=self.n_features * 2,
            out_features=self.n_classes,
            num_cells=self.num_cells,
            activation_class=nn.ReLU,
            dropout=self.dropout,
        )

    @override
    def forward(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Logits:
        # Concatenate the masked features and the feature mask
        x = torch.cat((masked_features, feature_mask), dim=1)
        logits = self.classifier(x)
        return logits

    @override
    def __call__(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Logits:
        return self.forward(masked_features, feature_mask)
