from typing import Tuple

import lightning as pl
import torch
import torch.nn.functional as F
from jaxtyping import Float, Shaped
from torch import Tensor, nn, optim
from torchrl.modules import MLP

from afa_rl.custom_types import (
    Classifier,
    Embedder,
    Embedding,
    Logits,
    PermutationInvariantEncoder,
)
from afa_rl.utils import get_feature_set, get_image_feature_set, mask_data
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
        self.mlp = MLP(input_size, num_classes, num_cells=num_cells)

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


class PermutationInvariantEncoder1D(PermutationInvariantEncoder):
    """
    A PermutationInvariantEncoder designed for 1D data.
    """

    def __init__(self, element_encoder: nn.Module):
        super().__init__()

        self.element_encoder = element_encoder  # h in the paper

    def forward(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Float[Tensor, "*batch latent_size"]:
        """
        Args:
            masked_features: currently observed features, with zeros for missing features
            feature_mask: indicator for missing features, 1 if feature is observed, 0 if missing
        Returns:
            code: the embedding of the input features
        """

        # Concatenate the feature values with their one-hot indices
        feature_set = get_feature_set(masked_features, feature_mask)

        # Pass through the element encoder
        encoded = self.element_encoder(
            feature_set
        )  # Shape: (batch_size, num_elements, latent_size)

        # Perform max pooling over the sequence dimension
        encoding = torch.max(encoded, dim=-2)[0]  # Shape: (batch_size, 1, latent_size)
        encoding = encoding.squeeze(-2)  # Shape: (batch_size, latent_size)

        return encoding


class PermutationInvariantEncoder2D(PermutationInvariantEncoder):
    """
    A PermutationInvariantEncoder designed for 2D data.
    """

    def __init__(self, element_encoder: nn.Module, image_shape: Tuple[int, int]):
        super().__init__()

        # element_encoder has to take inputs of shape (batch_size, num_elements, 3). The last dimension is (value, height, width)
        self.element_encoder = element_encoder  # h in the paper

        self.image_shape = image_shape  # (height, width)

    def forward(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Float[Tensor, "*batch latent_size"]:
        """
        Args:
            masked_features: currently observed features, with zeros for missing features
            feature_mask: indicator for missing features, 1 if feature is observed, 0 if missing
        Returns:
            code: the embedding of the input features
        """

        # Concatenate the feature values with image indices
        feature_set = get_image_feature_set(
            masked_features, feature_mask, image_shape=self.image_shape
        )  # Shape: (batch_size, num_elements, 3)

        # Pass through the element encoder
        encoded = self.element_encoder(
            feature_set
        )  # Shape: (batch_size, num_elements, latent_size)

        # Perform max pooling over the sequence dimension
        encoding = torch.max(encoded, dim=-2)[0]  # Shape: (batch_size, 1, latent_size)
        encoding = encoding.squeeze(-2)  # Shape: (batch_size, latent_size)

        return encoding


class PartialVAE(pl.LightningModule):
    """
    A partial VAE for masked data, as described in "EDDI: Efficient Dynamic Discovery of High-Value Information with Partial VAE"

    To make the model work with different shapes of data, change the encoder.
    """

    def __init__(
        self,
        encoder: PermutationInvariantEncoder,
        fc_mu: nn.Module,
        fc_logvar: nn.Module,
        decoder: nn.Module,
        lr=1e-3,
    ):
        """
        Args:
            encoder: the encoder to use. Should be a permutation invariant encoder.
            fc_mu: the MLP to use for the mean of the latent space
            fc_logvar: the MLP to use for the log variance of the latent space
            decoder: the MLP to use for the decoder
            lr: learning rate for the optimizer
        """
        super().__init__()

        self.encoder = encoder

        self.fc_mu = fc_mu
        self.fc_logvar = fc_logvar
        self.decoder = decoder  # Maps from latent space to the original feature space
        self.lr = lr

    def forward(self, masked_features: MaskedFeatures, feature_mask: FeatureMask):
        encoding = self.encoder(masked_features, feature_mask)

        mu, logvar = self.fc_mu(encoding), self.fc_logvar(encoding)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

    def training_step(self, batch, batch_idx):
        features, _ = batch  # ignore labels when training VAE for reconstruction
        masked_features, feature_mask = mask_data(features, p=0.5)
        estimated_features, mu, logvar = self(masked_features, feature_mask)
        loss = self.loss_function(estimated_features, features, mu, logvar)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        features, _ = batch  # ignore labels when training VAE for reconstruction
        masked_features, feature_mask = mask_data(features, p=0.5)
        estimated_features, mu, logvar = self(masked_features, feature_mask)
        loss = self.loss_function(estimated_features, features, mu, logvar)
        self.log("val_loss", loss)
        return loss

    def loss_function(self, estimated_features, features, mu, logvar):
        recon_loss = ((estimated_features - features) ** 2).sum()
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_div

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
