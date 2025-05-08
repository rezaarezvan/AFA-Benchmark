
from typing import Tuple

import lightning as pl
import torch
import torch.nn.functional as F
from jaxtyping import Float, Shaped
from torch import Tensor, nn, optim
from torchrl.modules import MLP

from afa_rl.custom_types import (
    MaskedClassifier,
    Logits,
    NNMaskedClassifier,
)
from afa_rl.shim2018.custom_types import Embedder, Embedding, EmbeddingClassifier
from afa_rl.utils import (
    get_feature_set,
)
from common.custom_types import FeatureMask, MaskedFeatures

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
            1, batch_size, 2 * self.memory_size, device=input_set.device
        )  # initial hidden state
        for _ in range(self.processing_steps):
            q: Float[Tensor, "{batch_size} 1 {2*self.memory_size}"] = self.rnn(
                torch.zeros(batch_size, 1, 1, device=input_set.device), h
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


class Shim2018EmbedderClassifier(pl.LightningModule):
    """
    A module that combines the ShimEncoder with a classifier for pretraining.
    """

    def __init__(self, embedder: Shim2018Embedder, classifier: Shim2018MLPClassifier, class_probabilities: Float[Tensor, "n_classes"], lr=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=["embedder", "classifier"])
        self.lr = lr
        self.embedder = embedder
        self.classifier = classifier
        self.class_weight = 1 / class_probabilities
        self.class_weight = self.class_weight / torch.sum(self.class_weight)

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

    def training_step(self, batch, batch_idx):
        feature_values, y = batch
        # Half of the samples will 50% probability of each feature being observed
        # The other half will have fully observed features
        feature_mask = torch.ones_like(feature_values, dtype=torch.bool, device=feature_values.device)
        feature_mask[: feature_mask.shape[0] // 2] = torch.randint(
            0, 2, feature_mask[: feature_mask.shape[0] // 2].shape
        )
        feature_values[feature_mask == 0] = 0
        _, y_hat = self(feature_values, feature_mask)
        loss = F.cross_entropy(y_hat, y, weight=self.class_weight.to(y_hat.device))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        feature_values, y = batch

        # Half of the samples will 50% probability of each feature being observed
        # The other half will have fully observed features
        feature_mask = torch.ones_like(feature_values, dtype=torch.bool, device=feature_values.device)
        n = feature_mask.shape[0] // 2
        feature_mask[:n] = torch.randint(
            0, 2, feature_mask[:n].shape
        )
        feature_values[feature_mask == 0] = 0

        _, y_hat = self(feature_values, feature_mask)
        loss = F.cross_entropy(y_hat, y, weight=self.class_weight.to(y_hat.device))
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


class Shim2018NNMaskedClassifier(NNMaskedClassifier):
    """A wrapper for the ShimEmbedderClassifier to make it compatible with the NNMaskedClassifier interface."""
    def __init__(self, embedder_and_classifier: Shim2018EmbedderClassifier):
        super().__init__()
        self.embedder_and_classifier = embedder_and_classifier

    def forward(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Logits:
        embedding, logits = self.embedder_and_classifier(masked_features, feature_mask)
        return logits

class Shim2018MaskedClassifier(MaskedClassifier):
    """A wrapper for the ShimEmbedderClassifier to make it compatible with the MaskedClassifier interface."""
    def __init__(self, embedder_and_classifier: Shim2018EmbedderClassifier):
        self.embedder_and_classifier: Shim2018EmbedderClassifier = embedder_and_classifier

    def __call__(self, masked_features: MaskedFeatures, feature_mask: FeatureMask) -> Logits:
        model_device = next(self.embedder_and_classifier.parameters()).device
        masked_features = masked_features.to(model_device)
        feature_mask = feature_mask.to(model_device)
        with torch.no_grad():
            embedding, logits = self.embedder_and_classifier.forward(masked_features, feature_mask)
        return logits
