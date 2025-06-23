from enum import Enum
from pathlib import Path
from typing import Self, final, override

import lightning as pl
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

from afa_rl.custom_types import (
    NaiveIdentityFn,
)
from afa_rl.utils import (
    mask_data,
)
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
    A PointNet(Plus).
    """

    def __init__(
        self,
        naive_identity_fn: NaiveIdentityFn,
        identity_network: nn.Module,
        feature_map_encoder: nn.Module,
        pointnet_type: PointNetType,
    ):
        """
        Args:
            naive_identity_fn: takes a FeatureMask as input and outputs a suitable location-like representation for each feature. Usually one-hot indices for 1D data and coordinates for 2D data.
            identity_network: transforms naive identity into a vector to be used with PNP
            feature_map_encoder: nn.Module
            pointnet_type: PointNetType, either POINTNET or POINTNETPLUS
        """
        super().__init__()

        self.naive_identity_fn = naive_identity_fn  # naive e
        self.identity_network = identity_network  # learns e from naive e
        self.feature_map_encoder = feature_map_encoder  # h in the paper
        self.pointnet_type = pointnet_type

    @override
    def forward(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Float[Tensor, "*batch pointnet_size"]:
        """
        Args:
            masked_features: currently observed features, with zeros for missing features
            feature_mask: indicator for missing features, 1 if feature is observed, 0 if missing
        Returns:
            the encoding of the input features
        """

        # Get coordinates of each feature (naive e)
        naive_identity = self.naive_identity_fn(
            feature_mask
        )  # Shape: (batch_size, n_features, naive_identity_size)

        # Pass it through the identity network to learn the identity
        identity = self.identity_network(
            naive_identity
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
        )  # Shape: (batch_size, element_encoding_size)

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
    def __init__(
        self,
        partial_vae: PartialVAE,
        classifier: nn.Module,
        class_probabilities: Float[Tensor, "n_classes"],
        min_masking_probability: float,
        max_masking_probability: float,
        lr: float,
        kl_scaling_factor: float = 1e-3,
        classifier_loss_scaling_factor: float = 1,
        recon_loss_type: PartialVAELossType = PartialVAELossType.SQUARED_ERROR,
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
        self.kl_scaling_factor: float = kl_scaling_factor

        self.recon_loss_type = recon_loss_type

    @override
    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int):
        features: Features = batch[0]
        label: Label = batch[1]

        masking_probability = self.min_masking_probability + torch.rand(1).item() * (
            self.max_masking_probability - self.min_masking_probability
        )
        self.log("masking_probability", masking_probability, sync_dist=True)

        masked_features, feature_mask, _ = mask_data(features, p=masking_probability)

        # Pass masked features through VAE, returning estimated features but also encoding which will be passed through classifier
        encoding, mu, logvar, z, estimated_features = self.partial_vae(
            masked_features, feature_mask
        )
        partial_vae_loss, partial_vae_recon_loss, partial_vae_kl_div_loss = (
            self.partial_vae_loss_function(estimated_features, features, mu, logvar)
        )
        self.log("train_loss_vae", partial_vae_loss, sync_dist=True)
        self.log("train_recon_loss_vae", partial_vae_recon_loss, sync_dist=True)
        self.log("train_kl_div_loss_vae", partial_vae_kl_div_loss, sync_dist=True)

        # Pass the encoding through the classifier
        logits = self.classifier(z)
        classifier_loss = F.cross_entropy(
            logits, label.float(), weight=self.class_weights.to(logits.device)
        )
        self.log("train_loss_classifier", classifier_loss, sync_dist=True)

        total_loss = (
            partial_vae_loss + self.classifier_loss_scaling_factor * classifier_loss
        )
        self.log("train_loss", total_loss, sync_dist=True)

        return total_loss

    def _get_loss_and_acc(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features: Features,
        label: Label,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # Pass masked features through VAE, returning estimated features but also encoding which will be passed through classifier
        _encoder, mu, logvar, z, estimated_features = self.partial_vae(
            masked_features, feature_mask
        )
        partial_vae_loss, partial_vae_recon_loss, partial_vae_kl_div_loss = (
            self.partial_vae_loss_function(estimated_features, features, mu, logvar)
        )

        # Pass the encoding through the classifier
        logits = self.classifier(z)
        classifier_loss = F.cross_entropy(
            logits, label, weight=self.class_weights.to(logits.device)
        )

        # For validation, additionally calculate accuracy
        y_pred = torch.argmax(logits, dim=1)
        y_cls = torch.argmax(label, dim=1)
        acc = (y_pred == y_cls).float().mean()

        return (
            partial_vae_loss,
            partial_vae_recon_loss,
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
        feature_values, y = batch

        # Mask features with minimum probability -> see many features (observations)
        feature_mask_many_observations = (
            torch.rand(feature_values.shape, device=feature_values.device)
            > self.min_masking_probability
        )
        feature_values_many_observations = feature_values.clone()
        feature_values_many_observations[feature_mask_many_observations == 0] = 0
        (
            loss_vae_many_observations,
            recon_loss_vae_many_observations,
            kl_div_loss_vae_many_observations,
            loss_classifier_many_observations,
            acc_many_observations,
        ) = self._get_loss_and_acc(
            feature_values_many_observations,
            feature_mask_many_observations,
            feature_values,
            y,
        )
        self.log("val_loss_vae_many_observations", loss_vae_many_observations)
        self.log(
            "val_recon_loss_vae_many_observations", recon_loss_vae_many_observations
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
        feature_mask_few_observations = (
            torch.rand(feature_values.shape, device=feature_values.device)
            > self.max_masking_probability
        )
        feature_values_few_observations = feature_values.clone()
        feature_values_few_observations[feature_mask_few_observations == 0] = 0
        (
            loss_vae_few_observations,
            recon_loss_vae_few_observations,
            kl_div_loss_vae_few_observations,
            loss_classifier_few_observations,
            acc_few_observations,
        ) = self._get_loss_and_acc(
            feature_values_few_observations,
            feature_mask_few_observations,
            feature_values,
            y,
        )
        self.log("val_loss_vae_few_observations", loss_vae_few_observations)
        self.log("val_recon_loss_vae_few_observations", recon_loss_vae_few_observations)
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
        self, estimated_features: Tensor, features: Tensor, mu: Tensor, logvar: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.recon_loss_type == PartialVAELossType.SQUARED_ERROR:
            recon_loss = ((estimated_features - features) ** 2).sum(dim=1).mean(dim=0)
        elif self.recon_loss_type == PartialVAELossType.BINARY_CROSS_ENTROPY:
            recon_loss = F.binary_cross_entropy(
                estimated_features, features, reduction="mean"
            )
        else:
            raise ValueError(
                f"Unknown reconstruction loss type: {self.recon_loss_type}. Use any of {set(map(str, PartialVAELossType))}"
            )
        kl_div = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean(dim=0)
        kl_div_loss = self.kl_scaling_factor * kl_div
        return recon_loss + kl_div_loss, recon_loss, kl_div_loss

    @override
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def generate_data(
        self, latent_size: int, device: torch.device, n_samples: int = 1
    ) -> tuple[Tensor, Tensor]:
        """Generate `n_samples` of new data.

        Args:
            - latent_size (int): the size of the latent space, needed for sampling
            - n_samples (int): how many samples to generate
            - device (int): where to place the sampled latent vectors before passing them to the model
        """

        dist = torch.distributions.MultivariateNormal(
            loc=torch.zeros(latent_size), covariance_matrix=torch.eye(latent_size)
        )
        z = dist.sample(torch.Size((n_samples,)))
        z = z.to(device)

        # Decode for features
        x_hat = self.partial_vae.decoder(z)

        # Apply classifier for class probabilities
        logits = self.classifier(z)
        probs = logits.softmax(dim=-1)

        return x_hat, probs


@final
class Zannone2019AFAPredictFn(AFAPredictFn):
    """A wrapper for the Zannone2019PretrainingModel to make it compatible with the AFAPredictFn interface."""

    def __init__(self, model: Zannone2019PretrainingModel):
        super().__init__()
        self.model = model

    @override
    def __call__(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Label:
        _encoding, _mu, _logvar, z = self.model.partial_vae.encode(
            masked_features, feature_mask
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

    @override
    def __call__(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Label:
        original_device = masked_features.device

        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)

        _encoding, _mu, _logvar, z = self.model.partial_vae.encode(
            masked_features, feature_mask
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
