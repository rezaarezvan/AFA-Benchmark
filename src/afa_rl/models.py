from enum import Enum
from typing import Tuple

import lightning as pl
import torch
import torch.nn.functional as F
from jaxtyping import Float, Shaped
from lightning.fabric.utilities import rank_zero_only
from matplotlib import pyplot as plt
from torch import Tensor, nn, optim
from torchrl.modules import MLP

import wandb
from afa_rl.custom_types import (
    Classifier,
    Embedder,
    Embedding,
    Label,
    Logits,
    NaiveIdentityFn,
    PointNetLike,
)
from afa_rl.utils import (
    get_feature_set,
    mask_data,
)
from common.custom_types import FeatureMask, Features, MaskedFeatures


class ReadProcessEncoder(pl.LightningModule):
    """
    A set encoder using RNNs, as described in the paper "Order Matters: Sequence to sequence for sets" (http://arxiv.org/abs/1511.06391).
    """

    def __init__(
        self,
        feature_size: int,  # size of each element in input sequence. Usually 1 (scalar sequence)
        output_size: int,  # size of output vector, one output per sequence
        reading_block_cells: list[int] = [32, 32],
        writing_block_cells: list[int] = [32, 32],
        memory_size: int = 16,  # each element in the sequence gets converted into a memory with this size
        processing_steps: int = 5,  # RNN processing steps. Paper shows that 5-10 is good.
        criterion: nn.Module = nn.MSELoss(),  # feel free to change this to any other loss function
        lr: float = 0.001,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["criterion"])

        self.feature_size = feature_size
        self.output_size = output_size
        self.reading_block_cells = reading_block_cells
        self.writing_block_cells = writing_block_cells
        self.memory_size = memory_size
        self.processing_steps = processing_steps
        self.lr = lr

        self.reading_block = MLP(
            in_features=feature_size,
            out_features=memory_size,
            num_cells=reading_block_cells,
        )
        # An RNN without input! The input will always be zero and the sequence length will be 1
        self.rnn = nn.GRU(input_size=1, hidden_size=2 * memory_size, batch_first=True)
        # The RNN output has to be projected to the memory size
        self.proj = nn.Linear(2 * memory_size, memory_size)

        # After the processing steps, the final memory is passed through a MLP to produce the output
        # The paper uses a pointer network but we only want a single feature vector
        self.write_block = MLP(
            in_features=2 * memory_size,
            out_features=output_size,
            num_cells=writing_block_cells,
        )

        self.criterion = criterion

    def forward(
        self, input_set: Shaped[Tensor, "N L {self.feature_size}"]
    ) -> Shaped[Tensor, "N {self.output_size}"]:
        batch_size = input_set.shape[0]
        seq_length = input_set.shape[1]  # noqa: F841, variable is used in array type hints
        memories: Float[Tensor, "{batch_size} {seq_length} {self.memory_size}"] = (
            self.reading_block(input_set)
        )
        h = torch.zeros(
            1, batch_size, 2 * self.memory_size, device=self.device
        )  # initial hidden state
        for _ in range(self.processing_steps):
            q: Float[Tensor, "{batch_size} 1 {2*self.memory_size}"] = self.rnn(
                torch.zeros(batch_size, 1, 1, device=self.device), h
            )[0]
            q: Float[Tensor, "{batch_size} {self.memory_size}"] = self.proj(
                q.squeeze(1)
            )
            # Take the dotproduct of the query with each memory
            e: Float[Tensor, "{batch_size} {seq_length}"] = torch.bmm(
                memories, q.unsqueeze(-1)
            ).squeeze(-1)
            # Softmax over sequence dimension
            a: Float[Tensor, "{batch_size} {seq_length}"] = torch.softmax(e, dim=-1)
            # Reshape a to (batch_size, 1, seq_length) for block matrix multiplication
            a: Float[Tensor, "{batch_size} 1 {seq_length}"] = a.unsqueeze(1)
            # Linear combination of memories. (1,seq_length) x (seq_length,memory_size) -> (1,memory_size)
            r: Float[Tensor, "{batch_size} 1 {self.memory_size}"] = torch.bmm(
                a, memories
            )
            r: Float[Tensor, "{batch_size} {self.memory_size}"] = r.squeeze(1)
            # Concatenate r with q to produce next long-term memory
            h: Float[Tensor, "{batch_size} {2*self.memory_size}"] = torch.cat(
                [q, r], dim=-1
            )
            # Conform to torch standard for RNNs
            h: Float[Tensor, "1 {batch_size} {2*self.memory_size}"] = h.unsqueeze(0)
        # Final h is the output of the process block
        h: Float[Tensor, "{batch_size} {2*self.memory_size}"] = h.squeeze(0)
        output: Float[Tensor, "{batch_size} {self.output_size}"] = self.write_block(h)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


class ShimEmbedder(Embedder):
    """
    Wraps a ReadProcessEmbedder to handle inputs consisting of features and their indices, using the representation
    described in "Joint Active Feature Acquisition and Classification with Variable-Size Set Encoding"
    """

    def __init__(self, encoder: ReadProcessEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Embedding:
        feature_set = get_feature_set(masked_features, feature_mask)
        return self.encoder(feature_set)


class MLPClassifier(Classifier):
    def __init__(self, input_size: int, num_classes: int, num_cells):
        super().__init__()
        self.mlp = MLP(
            input_size, num_classes, num_cells=num_cells, activation_class=nn.ReLU
        )

    def forward(self, embedding: Embedding) -> Logits:
        return self.mlp(embedding)


class ShimEmbedderClassifier(pl.LightningModule):
    """
    A module that combines the ShimEncoder with a classifier for pretraining.
    """

    def __init__(self, embedder: ShimEmbedder, classifier: MLPClassifier, lr=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=["embedder", "classifier"])
        self.lr = lr
        self.embedder = embedder
        self.classifier = classifier

    def forward(
        self, feature_values: MaskedFeatures, feature_mask: FeatureMask
    ) -> Tuple[Embedding, Logits]:
        """
        Args:
            x: currently observed features, with zeros for missing features
            z: indicator for missing features, 1 if feature is observed, 0 if missing
        Returns:
            embedding: the embedding of the input features
            classifier_output: the output of the classifier
        """
        embedding = self.embedder(feature_values, feature_mask)
        classifier_output = self.classifier(embedding)
        return embedding, classifier_output

    def training_step(self, batch, batch_idx):
        feature_values, y = batch
        # Half of the samples will 50% probability of each feature being observed
        # The other half will have fully observed features
        feature_mask = torch.ones_like(feature_values, dtype=torch.bool)
        feature_mask[: feature_mask.shape[0] // 2] = torch.randint(
            0, 2, feature_mask[: feature_mask.shape[0] // 2].shape, device=self.device
        )
        feature_values[feature_mask == 0] = 0
        _, y_hat = self(feature_values, feature_mask)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        feature_values, y = batch

        # Half of the samples will 50% probability of each feature being observed
        # The other half will have fully observed features
        feature_mask = torch.ones_like(feature_values, dtype=torch.bool)
        n = feature_mask.shape[0] // 2
        feature_mask[:n] = torch.randint(
            0, 2, feature_mask[:n].shape, device=self.device
        )
        feature_values[feature_mask == 0] = 0

        _, y_hat = self(feature_values, feature_mask)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)

        y_pred = torch.argmax(y_hat, dim=1)
        y_cls = torch.argmax(y, dim=1)

        # Calculate accuracy for the missing feature case
        acc = (y_pred[:n] == y_cls[:n]).float().mean()
        self.log("val_acc", acc)

        # Calculate accuracy for the full feature case
        acc_full = (y_pred[n:] == y_cls[n:]).float().mean()
        self.log("val_acc_full", acc_full)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class PointNetType(Enum):
    POINTNET = 1
    POINTNETPLUS = 2


class PointNet(PointNetLike):
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


class PartialVAE(nn.Module):
    """
    A partial VAE for masked data, as described in "EDDI: Efficient Dynamic Discovery of High-Value Information with Partial VAE"

    To make the model work with different shapes of data, change the pointnet.
    """

    def __init__(
        self,
        pointnet: PointNetLike,
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

    def encode(self, masked_features: MaskedFeatures, feature_mask: FeatureMask):
        pointnet_output = self.pointnet(masked_features, feature_mask)
        encoding = self.encoder(pointnet_output)

        mu = encoding[:, : encoding.shape[1] // 2]
        logvar = encoding[:, encoding.shape[1] // 2 :]
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

        return encoding, mu, logvar, z

    def forward(self, masked_features: MaskedFeatures, feature_mask: FeatureMask):
        # Encode the masked features
        encoding, mu, logvar, z = self.encode(masked_features, feature_mask)

        # Decode
        x_hat = self.decoder(z)

        return encoding, mu, logvar, z, x_hat


class Zannone2019PretrainingModel(pl.LightningModule):
    def __init__(
        self,
        partial_vae: PartialVAE,
        classifier: nn.Module,
        lr: float,
        max_masking_probability: float,
        validation_masking_probability=0.0,
        verbose=False,
        kl_scaling_factor=1e-3,
        image_shape=None,
        recon_loss_type="squared error",
    ):
        super().__init__()
        self.partial_vae = partial_vae
        self.classifier = classifier
        self.lr = lr
        self.max_masking_probability = max_masking_probability
        self.validation_masking_probability = validation_masking_probability
        self.verbose = verbose
        self.kl_scaling_factor = kl_scaling_factor

        # If image_shape is given, show reconstructed images at validation time
        self.image_shape = image_shape

        self.recon_loss_type = recon_loss_type
        if recon_loss_type not in ["squared error", "cross entropy"]:
            raise ValueError(
                f"Unknown reconstruction loss type: {self.recon_loss_type}. Use 'squared error' or 'cross entropy'."
            )

        # Initial masking probability
        self.masking_probability = 0.0

    def on_train_epoch_start(self):
        # Masking probability uniformly distributed between 0 and self.max_masking_probability
        self.masking_probability = torch.rand(1).item() * self.max_masking_probability
        self.log("masking_probability", self.masking_probability, sync_dist=True)

    def training_step(self, batch, batch_idx):
        features: Features = batch[0]
        label: Label = batch[1]

        masked_features, feature_mask = mask_data(features, p=self.masking_probability)

        # Pass masked features through VAE, returning estimated features but also encoding which will be passed through classifier
        encoding, mu, logvar, z, estimated_features = self.partial_vae(
            masked_features, feature_mask
        )
        partial_vae_loss = self.partial_vae_loss_function(
            estimated_features, features, mu, logvar
        )
        self.log("train_loss_vae", partial_vae_loss, sync_dist=True)

        # Pass the encoding through the classifier
        classifier_output = self.classifier(encoding)
        classifier_loss = F.cross_entropy(classifier_output, label.float())
        self.log("train_loss_classifier", classifier_loss, sync_dist=True)

        total_loss = partial_vae_loss + classifier_loss
        self.log("train_loss", total_loss, sync_dist=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        features: Features = batch[0]
        label: Label = batch[1]

        # Constant masking probability for validation
        masked_features, feature_mask = mask_data(
            features, p=self.validation_masking_probability
        )

        # Pass masked features through VAE, returning estimated features but also encoding which will be passed through classifier
        encoding, mu, logvar, z, estimated_features = self.partial_vae(
            masked_features, feature_mask
        )
        partial_vae_loss = self.partial_vae_loss_function(
            estimated_features, features, mu, logvar
        )
        self.log("val_loss_vae", partial_vae_loss, sync_dist=True)

        # Pass the encoding through the classifier
        logits = self.classifier(encoding)
        classifier_loss = F.cross_entropy(logits, label.float())
        self.log("val_loss_classifier", classifier_loss, sync_dist=True)

        # For validation, additionally calculate accuracy
        y_pred = torch.argmax(logits, dim=1)
        y_cls = torch.argmax(label, dim=1)
        acc = (y_pred == y_cls).float().mean()
        self.log("val_acc", acc, sync_dist=True)

        total_loss = partial_vae_loss + classifier_loss
        self.log("val_loss", total_loss, sync_dist=True)

        if self.verbose:
            # Log the total L2 norm of all parameters in the autoencoder
            norm_vae = torch.norm(
                torch.stack(
                    [
                        torch.norm(p.detach())
                        for p in self.partial_vae.parameters()
                        if p.requires_grad
                    ]
                )
            )
            self.log("norm_vae", norm_vae, sync_dist=True)

            # Log the total L2 norm of all parameters in the classifier
            norm_classifier = torch.norm(
                torch.stack(
                    [
                        torch.norm(p.detach())
                        for p in self.classifier.parameters()
                        if p.requires_grad
                    ]
                )
            )
            self.log("norm_classifier", norm_classifier, sync_dist=True)

        # If self.image_shape is defined, plot 4 images, their reconstructions and the predicted labels
        if batch_idx == 0:
            # If dataset consists of images, plot them
            if self.image_shape:
                self.log_val_images(
                    features=masked_features,
                    estimated_features=estimated_features,
                    z=z,
                    y_cls=y_cls,
                    y_pred=y_pred,
                )
            # Otherwise plot features as 1D signals
            else:
                self.log_val_features(
                    features=masked_features,
                    estimated_features=estimated_features,
                    z=z,
                    y_cls=y_cls,
                    y_pred=y_pred,
                )

        return total_loss

    @rank_zero_only
    def log_val_images(self, features, z, estimated_features, y_cls, y_pred):
        # Plot the first 4 images, their reconstructions and the predicted labels
        fig, axs = plt.subplots(3, 4, figsize=(20, 6))
        for i in range(4):
            axs[0, i].imshow(
                features[i].cpu().numpy().reshape(self.image_shape), cmap="gray"
            )
            axs[0, i].axis("off")
            axs[1, i].plot(z[i].cpu().numpy())
            axs[1, i].set_title("Latent space")
            axs[2, i].imshow(
                estimated_features[i].cpu().numpy().reshape(self.image_shape),
                cmap="gray",
            )
            axs[2, i].axis("off")
            # Labels as titles
            axs[0, i].set_title(f"True: {y_cls[i].item()}")
            axs[2, i].set_title(f"Pred: {y_pred[i].item()}")

        wandb.log({"val_recon": wandb.Image(fig)})
        plt.close(fig)

    @rank_zero_only
    def log_val_features(self, features, z, estimated_features, y_cls, y_pred):
        # Plot the first 4 samples
        fig, axs = plt.subplots(3, 4, figsize=(20, 10))
        for i in range(4):
            axs[0, i].plot(features[i].cpu().numpy())
            axs[1, i].plot(z[i].cpu().numpy())
            axs[2, i].plot(estimated_features[i].cpu().numpy())
            # Labels as titles
            axs[0, i].set_title(f"True: {y_cls[i].item()}")
            axs[1, i].set_title("Latent space")
            axs[2, i].set_title(f"Pred: {y_pred[i].item()}")

        wandb.log({"val_recon": wandb.Image(fig)})
        plt.close(fig)

    def partial_vae_loss_function(self, estimated_features, features, mu, logvar):
        if self.recon_loss_type == "squared error":
            recon_loss = ((estimated_features - features) ** 2).sum()
        elif self.recon_loss_type == "cross entropy":
            recon_loss = F.binary_cross_entropy(
                estimated_features, features, reduction="sum"
            )
        else:
            raise ValueError(
                f"Unknown reconstruction loss type: {self.recon_loss_type}. Use 'squared error' or 'cross entropy'."
            )
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.kl_scaling_factor * kl_div

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
