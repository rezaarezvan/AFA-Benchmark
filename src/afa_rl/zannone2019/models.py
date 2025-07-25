from enum import Enum
from pathlib import Path
from typing import Self, final, override

import lightning as pl
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

from afa_rl.utils import mask_data, weighted_cross_entropy
from common.custom_types import (
    AFAClassifier,
    AFAPredictFn,
    FeatureMask,
    Features,
    MaskedFeatures,
    Label,
)


class PointNetType(Enum):
    POINTNET = 1
    POINTNETPLUS = 2


@final
class PointNet(nn.Module):
    """
    Implements the PointNet and PointNetPlus architectures for encoding sets of features, as described in
    "EDDI: Efficient Dynamic Discovery of High-Value Information with Partial VAE".

    This module learns a per-feature identity embedding and combines it with observed feature values
    using either concatenation (PointNet) or pointwise multiplication (PointNetPlus). The resulting
    representations are passed through a feature map encoder and aggregated to produce a fixed-size encoding.

    Args:
        identity_size (int): Size of the identity embedding for each feature.
        n_features (int): Number of input features.
        feature_map_encoder (nn.Module): Module to encode per-feature representations.
        pointnet_type (PointNetType): Type of PointNet variant to use (POINTNET or POINTNETPLUS).
        max_embedding_norm (float | None, optional): Maximum norm for the identity embeddings.
    """

    def __init__(
        self,
        identity_size: int,
        n_features: int,
        feature_map_encoder: nn.Module,
        pointnet_type: PointNetType,
        max_embedding_norm: float | None = None,
    ):
        """
        Initializes the PointNet module.

        Args:
            identity_size (int): Size of the identity embedding for each feature.
            n_features (int): Number of input features.
            feature_map_encoder (nn.Module): Module to encode per-feature representations.
            pointnet_type (PointNetType): Type of PointNet variant to use (POINTNET or POINTNETPLUS).
            max_embedding_norm (float | None, optional): Maximum norm for the identity embeddings.
        """
        super().__init__()

        self.identity_size = identity_size
        self.n_features = n_features
        self.feature_map_encoder = feature_map_encoder  # h in the paper
        self.pointnet_type = pointnet_type
        self.max_embedding_norm = max_embedding_norm

        self.embedding_net = nn.Embedding(
            self.n_features, self.identity_size, max_norm=self.max_embedding_norm
        )

    @override
    def forward(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Float[Tensor, "*batch pointnet_size"]:
        """
        Encodes a batch of masked feature vectors using PointNet or PointNetPlus.

        Args:
            masked_features (MaskedFeatures):
                Tensor of observed feature values, with zeros for missing features.
                Shape: (batch_size, n_features)
            feature_mask (FeatureMask):
                Binary mask indicating observed features (1 if observed, 0 if missing).
                Shape: (batch_size, n_features)

        Returns:
            Float[Tensor, "*batch pointnet_size"]:
                Encoded representation of the input features.
                Shape: (batch_size, feature_map_size)
        """

        assert masked_features.shape[1] == self.n_features
        assert feature_mask.shape[1] == self.n_features

        # Identity is a learnable embedding according to EDDI paper
        identity = self.embedding_net(
            torch.arange(
                masked_features.shape[-1], device=masked_features.device
            ).repeat(masked_features.shape[0], 1)
        )  # Shape: (batch_size, n_features, identity_size)

        # Could not think of a better name than s...
        if self.pointnet_type == PointNetType.POINTNETPLUS:
            # PointNetPlus does pointwise-multiplication between each identity vector and feature value
            s = (
                masked_features.unsqueeze(-1) * identity
            )  # Shape: (batch_size, n_features, identity_size)
        elif self.pointnet_type == PointNetType.POINTNET:
            # Normal PointNet concatenates feature value with identity vector
            s = torch.cat(
                [masked_features.unsqueeze(-1), identity], dim=-1
            )  # Shape: (batch_size, n_features, identity_size + 1)
        else:
            raise ValueError(f"Unknown PointNet type: {self.pointnet_type}")

        # Pass s through the feature map encoder (h)
        feature_maps = self.feature_map_encoder(
            s
        )  # Shape: (batch_size, n_features, feature_map_size)

        # Mask out the unobserved features with zeros
        feature_maps = feature_maps * feature_mask.unsqueeze(
            -1
        )  # Shape: (batch_size, n_features, feature_map_size)

        # Sum over n_features dimension
        encoding = torch.sum(
            feature_maps, dim=-2
        )  # Shape: (batch_size, feature_map_size)

        return encoding


@final
class PartialVAE(nn.Module):
    """A partial VAE for masked data, as described in "EDDI: Efficient Dynamic Discovery of High-Value Information with Partial VAE".

    To make the model work with different shapes of data, change the pointnet.
    """

    def __init__(
        self,
        pointnet: PointNet,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        """
        Args:
            pointnet: maps unordered sets of features to a single vector
            encoder: a network that maps the output from the pointnet to input for mu_net and logvar_net
            decoder: the network to use for the decoder
        """
        super().__init__()

        self.pointnet = pointnet
        self.encoder = encoder
        self.decoder = decoder  # Maps from latent space to the original feature space

    def encode(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
    ):
        pointnet_output = self.pointnet(masked_features, feature_mask)
        encoding = self.encoder(pointnet_output)

        mu = encoding[:, : encoding.shape[1] // 2]
        logvar = encoding[:, encoding.shape[1] // 2 :]
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

        return encoding, mu, logvar, z

    @override
    def forward(self, masked_features: MaskedFeatures, feature_mask: FeatureMask):
        # Encode the masked features
        encoding, mu, logvar, z = self.encode(masked_features, feature_mask)

        # Decode
        x_hat = self.decoder(z)

        return encoding, mu, logvar, z, x_hat


class PartialVAELossType(Enum):
    SQUARED_ERROR = 1
    BINARY_CROSS_ENTROPY = 2


@final
class Zannone2019PretrainingModel(pl.LightningModule):
    """Training the PartialVAE model as described in the paper "ODIN: Optimal Discovery of High-value INformation Using Model-based Deep Reinforcement Learning". This means that labels are appended to the masked feature vector, creating "augmented" features."""

    def __init__(
        self,
        partial_vae: PartialVAE,
        classifier: nn.Module,
        class_probabilities: Float[Tensor, "n_classes"],
        min_masking_probability: float,
        max_masking_probability: float,
        lr: float,
        start_kl_scaling_factor: float,
        end_kl_scaling_factor: float,
        n_annealing_epochs: int,
        classifier_loss_scaling_factor: float,  # how much more to weigh the classifier's loss compared to the PVAE's loss
        reconstruct_label: bool,
        label_loss_scaling_factor: float,  # how much more to weigh the label reconstruction loss compared to the feature reconstruction loss
    ):
        super().__init__()
        self.partial_vae: PartialVAE = partial_vae
        self.classifier: nn.Module = classifier
        self.lr: float = lr
        self.min_masking_probability: float = min_masking_probability
        self.max_masking_probability: float = max_masking_probability
        self.class_weights = 1 / class_probabilities
        self.class_weights: Tensor = self.class_weights / torch.sum(self.class_weights)
        self.classifier_loss_scaling_factor = classifier_loss_scaling_factor
        self.start_kl_scaling_factor: float = start_kl_scaling_factor
        self.end_kl_scaling_factor: float = end_kl_scaling_factor
        self.n_annealing_epochs = n_annealing_epochs
        self.reconstruct_label = reconstruct_label
        self.label_loss_scaling_factor: float = label_loss_scaling_factor

        # self.recon_loss_type = recon_loss_type

    def current_kl_weight(self) -> float:
        """Compute the current KL weight using linear annealing."""
        progress = min(1.0, self.current_epoch / self.n_annealing_epochs)
        return (
            self.start_kl_scaling_factor
            + (self.end_kl_scaling_factor - self.start_kl_scaling_factor) * progress
        )

    @override
    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int):
        features: Features = batch[0]
        label: Label = batch[1]

        if self.reconstruct_label:
            # According to the paper, labels are appended to the features. "Augmented" = features + labels
            augmented_features = torch.cat(
                [features, label], dim=-1
            )  # (batch_size, n_features+n_classes)
        else:
            augmented_features = features

        masking_probability = self.min_masking_probability + torch.rand(1).item() * (
            self.max_masking_probability - self.min_masking_probability
        )
        self.log("masking_probability", masking_probability, sync_dist=True)

        augmented_masked_features, augmented_feature_mask, _ = mask_data(
            augmented_features, p=masking_probability
        )

        # Pass masked features through VAE, returning estimated features but also encoding which will be passed through classifier
        _encoding, mu, logvar, z, estimated_augmented_features = self.partial_vae(
            augmented_masked_features, augmented_feature_mask
        )

        # Log norm of augmented features and their reconstructions
        self.log(
            "augmented_features_norm",
            (augmented_features**2).sum(dim=-1).mean(dim=0).sqrt(),
        )
        self.log(
            "estimated_augmented_features_norm",
            (estimated_augmented_features**2).sum(dim=-1).mean(dim=0).sqrt(),
        )

        (
            partial_vae_loss,
            partial_vae_feature_recon_loss,
            partial_vae_label_recon_loss,
            partial_vae_kl_div_loss,
        ) = self.partial_vae_loss_function(
            estimated_augmented_features, augmented_features, mu, logvar
        )
        self.log("train_loss_vae", partial_vae_loss, sync_dist=True)
        self.log(
            "train_feature_recon_loss_vae",
            partial_vae_feature_recon_loss,
            sync_dist=True,
        )
        self.log(
            "train_label_recon_loss_vae", partial_vae_label_recon_loss, sync_dist=True
        )
        self.log("train_kl_div_loss_vae", partial_vae_kl_div_loss, sync_dist=True)

        # Pass the encoding through the classifier
        logits = self.classifier(z)
        classifier_loss = F.cross_entropy(
            logits, label.float(), weight=self.class_weights.to(logits.device)
        )
        classifier_loss = classifier_loss * self.classifier_loss_scaling_factor
        self.log("train_loss_classifier", classifier_loss, sync_dist=True)

        total_loss = partial_vae_loss + classifier_loss
        self.log("train_loss", total_loss, sync_dist=True)

        return total_loss

    @override
    def on_train_epoch_end(self) -> None:
        self.log("kl_weight", self.current_kl_weight())

    def _get_loss_and_acc(
        self,
        augmented_masked_features: MaskedFeatures,
        augmented_feature_mask: FeatureMask,
        augmented_features: Features,
        label: Label,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        # Pass masked features through VAE, returning estimated augmented features but also encoding which will be passed through classifier
        _encoder, mu, logvar, z, estimated_augmented_features = self.partial_vae(
            augmented_masked_features, augmented_feature_mask
        )
        (
            partial_vae_loss,
            partial_vae_feature_recon_loss,
            partial_vae_label_recon_loss,
            partial_vae_kl_div_loss,
        ) = self.partial_vae_loss_function(
            estimated_augmented_features, augmented_features, mu, logvar
        )

        # Pass the encoding through the classifier
        logits = self.classifier(z)
        classifier_loss = F.cross_entropy(
            logits, label, weight=self.class_weights.to(logits.device)
        )
        classifier_loss = classifier_loss * self.classifier_loss_scaling_factor

        # For validation, additionally calculate accuracy
        y_pred = torch.argmax(logits, dim=1)
        y_cls = torch.argmax(label, dim=1)
        acc = (y_pred == y_cls).float().mean()

        return (
            partial_vae_loss,
            partial_vae_feature_recon_loss,
            partial_vae_label_recon_loss,
            partial_vae_kl_div_loss,
            classifier_loss,
            acc,
        )

    # def verbose_log(self):
    #     # Log the total L2 norm of all parameters in the autoencoder
    #     norm_vae = torch.norm(
    #         torch.stack(
    #             [
    #                 torch.norm(p.detach())
    #                 for p in self.partial_vae.parameters()
    #                 if p.requires_grad
    #             ]
    #         )
    #     )
    #     self.log("norm_vae", norm_vae, sync_dist=True)
    #
    #     # Log the total L2 norm of all parameters in the classifier
    #     norm_classifier = torch.norm(
    #         torch.stack(
    #             [
    #                 torch.norm(p.detach())
    #                 for p in self.classifier.parameters()
    #                 if p.requires_grad
    #             ]
    #         )
    #     )
    #     self.log("norm_classifier", norm_classifier, sync_dist=True)
    #
    #     # If self.image_shape is defined, plot 4 images, their reconstructions and the predicted labels
    #     if batch_idx == 0:
    #         self.log_val_features(
    #             features=masked_features,
    #             estimated_features=estimated_features,
    #             z=z,
    #             y_cls=y_cls,
    #             y_pred=y_pred,
    #         )

    @override
    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int):
        features, label = batch
        if self.reconstruct_label:
            augmented_features = torch.cat(
                [features, label], dim=-1
            )  # (batch_size, n_features+n_classes)
        else:
            augmented_features = features

        # Mask features with minimum probability -> see many features (observations)
        augmented_feature_mask_many_observations = (
            torch.rand(augmented_features.shape, device=augmented_features.device)
            > self.min_masking_probability
        )
        augmented_masked_features_many_observations = augmented_features.clone()
        augmented_masked_features_many_observations[
            augmented_feature_mask_many_observations == 0
        ] = 0
        (
            loss_vae_many_observations,
            feature_recon_loss_vae_many_observations,
            label_recon_loss_vae_many_observations,
            kl_div_loss_vae_many_observations,
            loss_classifier_many_observations,
            acc_many_observations,
        ) = self._get_loss_and_acc(
            augmented_masked_features_many_observations,
            augmented_feature_mask_many_observations,
            augmented_features,
            label,
        )
        self.log("val_loss_vae_many_observations", loss_vae_many_observations)
        self.log(
            "val_feature_recon_loss_vae_many_observations",
            feature_recon_loss_vae_many_observations,
        )
        self.log(
            "val_label_recon_loss_vae_many_observations",
            label_recon_loss_vae_many_observations,
        )
        self.log(
            "val_kl_div_loss_vae_many_observations", kl_div_loss_vae_many_observations
        )
        self.log(
            "val_loss_classifier_many_observations", loss_classifier_many_observations
        )
        self.log(
            "val_loss_many_observations",
            loss_vae_many_observations + loss_classifier_many_observations,
        )
        self.log("val_acc_many_observations", acc_many_observations)

        # Mask features with maximum probability -> see few features (observations)
        augmented_feature_mask_few_observations = (
            torch.rand(augmented_features.shape, device=augmented_features.device)
            > self.max_masking_probability
        )
        augmented_masked_features_few_observations = augmented_features.clone()
        augmented_masked_features_few_observations[
            augmented_feature_mask_few_observations == 0
        ] = 0
        (
            loss_vae_few_observations,
            feature_recon_loss_vae_few_observations,
            label_recon_loss_vae_few_observations,
            kl_div_loss_vae_few_observations,
            loss_classifier_few_observations,
            acc_few_observations,
        ) = self._get_loss_and_acc(
            augmented_masked_features_few_observations,
            augmented_feature_mask_few_observations,
            augmented_features,
            label,
        )
        self.log("val_loss_vae_few_observations", loss_vae_few_observations)
        self.log(
            "val_feature_recon_loss_vae_few_observations",
            feature_recon_loss_vae_few_observations,
        )
        self.log(
            "val_label_recon_loss_vae_few_observations",
            label_recon_loss_vae_few_observations,
        )
        self.log(
            "val_kl_div_loss_vae_few_observations", kl_div_loss_vae_few_observations
        )
        self.log(
            "val_loss_classifier_few_observations", loss_classifier_few_observations
        )
        self.log(
            "val_loss_few_observations",
            loss_vae_few_observations + loss_classifier_few_observations,
        )
        self.log("val_acc_few_observations", acc_few_observations)

        # if self.verbose:
        #     self.verbose_log()

    # @rank_zero_only
    # def log_val_features(self, features, z, estimated_features, y_cls, y_pred):
    #     # Plot the first 4 samples
    #     fig, axs = plt.subplots(3, 4, figsize=(20, 10))
    #     for i in range(4):
    #         axs[0, i].plot(features[i].cpu().numpy())
    #         axs[1, i].plot(z[i].cpu().numpy())
    #         axs[2, i].plot(estimated_features[i].cpu().numpy())
    #         # Labels as titles
    #         axs[0, i].set_title(f"True: {y_cls[i].item()}")
    #         axs[1, i].set_title("Latent space")
    #         axs[2, i].set_title(f"Pred: {y_pred[i].item()}")
    #
    #     wandb.log({"val_recon": wandb.Image(fig)})
    #     plt.close(fig)

    def partial_vae_loss_function(
        self,
        estimated_augmented_features: Tensor,
        augmented_features: Tensor,
        mu: Tensor,
        logvar: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if self.reconstruct_label:
            # Split augmented features (features+labels) into their corresponding parts so we can have different losses for them
            features, labels = (
                augmented_features[..., : -len(self.class_weights)],
                augmented_features[..., -len(self.class_weights) :],
            )
            estimated_features, estimated_label_logits = (
                estimated_augmented_features[..., : -len(self.class_weights)],
                estimated_augmented_features[..., -len(self.class_weights) :],
            )
            label_recon_loss = F.cross_entropy(
                estimated_label_logits,
                labels,
                weight=self.class_weights.to(estimated_label_logits.device),
                reduction="mean",
            )
            label_recon_loss = label_recon_loss * self.label_loss_scaling_factor
        else:
            features = augmented_features
            estimated_features = estimated_augmented_features
            label_recon_loss = torch.tensor(
                [0.0], device=estimated_augmented_features.device
            )
        feature_recon_loss = (
            ((estimated_features - features) ** 2).sum(dim=1).mean(dim=0)
        )
        kl_div_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean(
            dim=0
        )
        kl_div_loss = kl_div_loss * self.current_kl_weight()
        return (
            feature_recon_loss + label_recon_loss + kl_div_loss,
            feature_recon_loss,
            label_recon_loss,
            kl_div_loss,
        )

    @override
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def generate_data(
        self,
        latent_size: int,
        n_features: int,
        device: torch.device,
        n_samples: int = 1,
    ) -> tuple[Tensor, Tensor]:
        """Generate `n_samples` of new data.

        Args:
            - latent_size (int): the size of the latent space, needed for sampling
            - n_features (int): How many features each sample contains. This is necessary to separate features from labels since the PVAE is trained with labels appended to the feature vector.
            - n_samples (int): how many samples to generate
            - device (int): where to place the sampled latent vectors before passing them to the model
        """

        dist = torch.distributions.MultivariateNormal(
            loc=torch.zeros(latent_size), covariance_matrix=torch.eye(latent_size)
        )
        z = dist.sample(torch.Size((n_samples,)))
        z = z.to(device)

        # Decode for features
        estimated_augmented_features = self.partial_vae.decoder(
            z
        )  # (batch_size, n_features+n_classes)

        if self.reconstruct_label:
            # Only the first n_features values are "real" features
            estimated_features = estimated_augmented_features[
                :, :n_features
            ]  # (batch_size, n_features)
        else:
            estimated_features = estimated_augmented_features

        # Apply classifier for class probabilities
        logits = self.classifier(z)
        probs = logits.softmax(dim=-1)

        return estimated_features, probs

    def fully_observed_reconstruction(
        self,
        features: Features,
        n_classes: int,
        label: Label | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct a sample by providing all features. Optionally provide the label as well"""
        return self.masked_reconstruction(
            masked_features=features,
            feature_mask=torch.ones(
                features.shape, dtype=torch.bool, device=features.device
            ),
            n_classes=n_classes,
            label=label,
        )

    def masked_reconstruction(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        n_classes: int,
        label: Label | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct a sample by providing masked features. Optionally provide the label as well"""
        assert label is None or self.reconstruct_label, (
            "label should not be provided if the model has not been trained to reconstruct labels"
        )

        if self.reconstruct_label:
            if label is None:
                label = torch.zeros(
                    masked_features.shape[0],
                    n_classes,
                    dtype=torch.float32,
                    device=masked_features.device,
                )
                label_mask = torch.full_like(label, False)
            else:
                label_mask = torch.full_like(label, True)

            augmented_masked_features = torch.cat(
                [masked_features, label], dim=-1
            )  # (batch_size, n_features+n_classes)
            augmented_feature_mask = torch.cat([feature_mask, label_mask], dim=-1)
        else:
            augmented_masked_features = masked_features
            augmented_feature_mask = feature_mask

        _encoder, _mu, _logvar, z, estimated_augmented_features = self.partial_vae(
            augmented_masked_features, augmented_feature_mask
        )
        if self.reconstruct_label:
            estimated_features = estimated_augmented_features[:, :-n_classes]
        else:
            estimated_features = estimated_augmented_features

        return z, estimated_features


@final
class Zannone2019AFAPredictFn(AFAPredictFn):
    """A wrapper for the Zannone2019PretrainingModel to make it compatible with the AFAPredictFn interface."""

    def __init__(self, model: Zannone2019PretrainingModel):
        super().__init__()
        self.model = model
        self.n_classes = len(model.class_weights)

    @override
    def __call__(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features: Features | None,
        label: Label | None,
    ) -> Label:
        batch_size = masked_features.shape[0]
        if self.model.reconstruct_label:
            augmented_masked_features = torch.cat(
                [
                    masked_features,
                    torch.zeros(
                        batch_size, self.n_classes, device=masked_features.device
                    ),
                ],
                dim=-1,
            )
            augmented_feature_mask = torch.cat(
                [
                    feature_mask,
                    torch.full(
                        (batch_size, self.n_classes), False, device=feature_mask.device
                    ),
                ],
                dim=-1,
            )
        else:
            augmented_masked_features = masked_features
            augmented_feature_mask = feature_mask
        _encoding, _mu, _logvar, z = self.model.partial_vae.encode(
            augmented_masked_features, augmented_feature_mask
        )
        logits = self.model.classifier(z)
        return logits.softmax(dim=-1)


@final
class Zannone2019AFAClassifier(AFAClassifier):
    """A wrapper for the Zannone2019PretrainingModel to make it compatible with the AFAClassifier interface."""

    def __init__(
        self,
        model: Zannone2019PretrainingModel,
        device: torch.device,
    ):
        super().__init__()
        self._device = device
        self.model = model.to(self._device)
        self.n_classes = len(model.class_weights)

    @override
    def __call__(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features: Features | None,
        label: Label | None,
    ) -> Label:
        original_device = masked_features.device

        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)

        batch_size = masked_features.shape[0]
        if self.model.reconstruct_label:
            augmented_masked_features = torch.cat(
                [
                    masked_features,
                    torch.zeros(
                        batch_size, self.n_classes, device=masked_features.device
                    ),
                ],
                dim=-1,
            )
            augmented_feature_mask = torch.cat(
                [
                    feature_mask,
                    torch.full(
                        (batch_size, self.n_classes), False, device=feature_mask.device
                    ),
                ],
                dim=-1,
            )
        else:
            augmented_masked_features = masked_features
            augmented_feature_mask = feature_mask

        _encoding, _mu, _logvar, z = self.model.partial_vae.encode(
            augmented_masked_features, augmented_feature_mask
        )
        logits = self.model.classifier(z)
        return logits.softmax(dim=-1).to(original_device)

    @override
    def save(self, path: Path) -> None:
        torch.save(self.model.cpu(), path)

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        model = torch.load(path, weights_only=False, map_location=device)
        return cls(model, device)

    @override
    def to(self, device: torch.device) -> Self:
        self.model = self.model.to(device)
        return self

    @property
    @override
    def device(self) -> torch.device:
        return self._device
