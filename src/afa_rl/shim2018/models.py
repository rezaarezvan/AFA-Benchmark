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


class ReadProcessEncoder(nn.Module):
    """A set encoder using RNNs, as described in the paper "Order Matters: Sequence to sequence for sets" (http://arxiv.org/abs/1511.06391)."""

    def __init__(
        self,
        feature_size: int,  # size of each element in input sequence. Usually 1 (scalar sequence)
        output_size: int,  # size of output vector, one output per sequence
        reading_block_cells: list[int] = [32, 32],
        writing_block_cells: list[int] = [32, 32],
        memory_size: int = 16,  # each element in the sequence gets converted into a memory with this size
        processing_steps: int = 5,  # RNN processing steps. Paper shows that 5-10 is good.
    ):
        super().__init__()
        self.feature_size = feature_size
        self.output_size = output_size
        self.reading_block_cells = reading_block_cells
        self.writing_block_cells = writing_block_cells
        self.memory_size = memory_size
        self.processing_steps = processing_steps

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

    def forward(self, input_set: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input_set: a tensor of shape (batch_size, seq_length, feature_size)

        Returns:
            output: a tensor of shape (batch_size, output_size)

        """
        batch_size = input_set.shape[0]
        memories = self.reading_block(
            input_set
        )  # (batch_size, seq_length, memory_size)
        h = torch.zeros(
            1, batch_size, 2 * self.memory_size, device=input_set.device
        )  # initial hidden state
        for _ in range(self.processing_steps):
            q = self.rnn(torch.zeros(batch_size, 1, 1, device=input_set.device), h)[
                0
            ]  # (batch_size, 1, 2 * memory_size)
            q = self.proj(q.squeeze(1))  # (batch_size, memory_size)
            # Take the dotproduct of the query with each memory
            e = torch.bmm(memories, q.unsqueeze(-1)).squeeze(
                -1
            )  # (batch_size, seq_length)
            # Softmax over sequence dimension
            a = torch.softmax(e, dim=-1)  # (batch_size, seq_length)
            # Reshape a to (batch_size, 1, seq_length) for block matrix multiplication
            a = a.unsqueeze(1)  # (batch_size, 1, seq_length)
            # Linear combination of memories. (1,seq_length) x (seq_length,memory_size) -> (1,memory_size)
            r = torch.bmm(a, memories)  # (batch_size, 1, memory_size)
            r = r.squeeze(1)  # (batch_size, memory_size)
            # Concatenate r with q to produce next long-term memory
            h = torch.cat([q, r], dim=-1)  # (batch_size, 2 * memory_size)
            # Conform to torch standard for RNNs
            h = h.unsqueeze(0)  # (1, batch_size, 2 * memory_size)
        # Final h is the output of the process block
        h = h.squeeze(0)  # (batch_size, 2 * memory_size)
        output = self.write_block(h)  # (batch_size, output_size)
        return output



class LitReadProcessEncoder(pl.LightningModule):
    def __init__(self, read_process_encoder: ReadProcessEncoder, lr: float = 1e-3, criterion=F.cross_entropy):
        super().__init__()
        self.save_hyperparameters(ignore=["read_process_encoder"])
        self.read_process_encoder = read_process_encoder
        self.lr = lr
        self.criterion = criterion

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


class Shim2018Embedder(Embedder):
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


class Shim2018MLPClassifier(EmbeddingClassifier):
    def __init__(self, input_size: int, num_classes: int, num_cells):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_cells = num_cells

        self.mlp = MLP(
            input_size, num_classes, num_cells=num_cells, activation_class=nn.ReLU
        )

    def forward(self, embedding: Embedding) -> Logits:
        return self.mlp(embedding)


class LitShim2018EmbedderClassifier(pl.LightningModule):
    """
    A module that combines the ShimEncoder with a classifier for pretraining.
    """

    def __init__(
        self,
        embedder: Shim2018Embedder,
        classifier: Shim2018MLPClassifier,
        class_probabilities: Float[Tensor, "n_classes"],
        max_masking_probability=1.0,
        lr=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["embedder", "classifier"])
        self.lr = lr
        self.embedder = embedder
        self.classifier = classifier
        self.class_weight = 1 / class_probabilities
        self.class_weight = self.class_weight / torch.sum(self.class_weight)
        self.max_masking_probability = max_masking_probability

        # Initial masking probability
        self.masking_probability = 0.0

    def forward(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Tuple[Embedding, Logits]:
        """
        Args:
            x: currently observed features, with zeros for missing features
            z: indicator for missing features, 1 if feature is observed, 0 if missing
        Returns:
            embedding: the embedding of the input features
            classifier_output: the output of the classifier
        """
        embedding = self.embedder(masked_features, feature_mask)
        classifier_output = self.classifier(embedding)
        return embedding, classifier_output

    def on_train_epoch_start(self):
        # Masking probability uniformly distributed between 0 and self.max_masking_probability
        self.masking_probability = torch.rand(1).item() * self.max_masking_probability
        self.log("masking_probability", self.masking_probability, sync_dist=True)

    def training_step(self, batch, batch_idx):
        features: Features = batch[0]
        label: Label = batch[1]

        # WARNING: this block is only valid for AFAContextDataset
        feature_mask_optimal = torch.zeros_like(
            features, dtype=torch.bool, device=features.device
        )
        feature_mask_optimal[:, 0] = 1
        for i in range(feature_mask_optimal.shape[0]):
            context = features[i, 0].int().item()
            feature_mask_optimal[i, 1 + context * 3 : 4 + context * 3] = 1
        feature_values_optimal = features.clone()
        feature_values_optimal[feature_mask_optimal == 0] = 0
        loss_optimal, acc_optimal = self._get_loss_and_acc(
            feature_values_optimal, feature_mask_optimal, label
        )
        self.log("train_loss_optimal", loss_optimal)
        self.log("train_acc_optimal", acc_optimal)
        # masked_features = feature_values_optimal
        # feature_mask = feature_mask_optimal

        masked_features, feature_mask = mask_data(features, p=self.masking_probability)

        _, y_hat = self(masked_features, feature_mask)
        loss = F.cross_entropy(y_hat, label, weight=self.class_weight.to(y_hat.device))
        self.log("train_loss", loss)
        # loss_full, acc_full = self._get_loss_and_acc(
        #     features, torch.ones_like(features, dtype=torch.bool, device=features.device), label
        # )
        return loss

    def _get_loss_and_acc(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask, y: Tensor
    ) -> tuple[Tensor, Tensor]:
        _, y_hat = self(masked_features, feature_mask)
        loss = F.cross_entropy(y_hat, y, weight=self.class_weight.to(y_hat.device))
        predicted_class = torch.argmax(y_hat, dim=1)
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


class Shim2018NNMaskedClassifier(NNMaskedClassifier):
    """A wrapper for the ShimEmbedderClassifier to make it compatible with the NNMaskedClassifier interface."""

    def __init__(self, embedder_and_classifier: LitShim2018EmbedderClassifier):
        super().__init__()
        self.embedder_and_classifier = embedder_and_classifier

    def forward(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Logits:
        embedding, logits = self.embedder_and_classifier(masked_features, feature_mask)
        return logits


class Shim2018MaskedClassifier(MaskedClassifier):
    """A wrapper for the ShimEmbedderClassifier to make it compatible with the MaskedClassifier interface."""

    def __init__(self, embedder_and_classifier: LitShim2018EmbedderClassifier):
        self.embedder_and_classifier: LitShim2018EmbedderClassifier = (
            embedder_and_classifier
        )

    def __call__(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Logits:
        model_device = next(self.embedder_and_classifier.parameters()).device
        masked_features = masked_features.to(model_device)
        feature_mask = feature_mask.to(model_device)
        with torch.no_grad():
            embedding, logits = self.embedder_and_classifier.forward(
                masked_features, feature_mask
            )
        return logits
