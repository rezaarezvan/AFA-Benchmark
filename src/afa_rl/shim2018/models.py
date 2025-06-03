from pathlib import Path
from typing import Self, final, override

import lightning as pl
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn
from torchrl.modules import MLP
from afa_rl.shim2018.custom_types import Embedder, Embedding, EmbeddingClassifier
from afa_rl.utils import (
    get_feature_set,
    mask_data,
    shuffle_feature_set,
)
from common.custom_types import (
    AFAClassifier,
    AFAPredictFn,
    FeatureMask,
    MaskedFeatures,
    Features,
    Label,
    Logits,
)
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pad_packed_sequence as pad


@final
class ReadProcessEncoder(nn.Module):
    """A set encoder using RNNs, as described in the paper "Order Matters: Sequence to sequence for sets" (http://arxiv.org/abs/1511.06391)."""

    def __init__(
        self,
        set_element_size: int,  # size of each element in input set. Usually 1 (scalar sequence)
        output_size: int,  # size of output vector, one output per sequence
        reading_block_cells: tuple[int, ...] = (32, 32),
        writing_block_cells: tuple[int, ...] = (32, 32),
        memory_size: int = 16,  # each element in the sequence gets converted into a memory with this size
        processing_steps: int = 5,  # RNN processing steps. Paper shows that 5-10 is good.
        dropout: float = 0.0,
    ):
        super().__init__()
        self.set_element_size = set_element_size
        self.output_size = output_size
        self.reading_block_cells = reading_block_cells
        self.writing_block_cells = writing_block_cells
        self.memory_size = memory_size
        self.processing_steps = processing_steps
        self.dropout = dropout

        self.reading_block = MLP(
            in_features=set_element_size,
            out_features=memory_size,
            num_cells=reading_block_cells,
            activation_class=nn.ReLU,
            dropout=dropout,
        )

        self.process_lstm = nn.LSTMCell(memory_size, memory_size)

        # After the processing steps, the final memory is passed through a MLP to produce the output
        # The paper uses a pointer network but we only want a single feature vector
        self.write_block = MLP(
            in_features=2 * memory_size,
            out_features=output_size,
            num_cells=writing_block_cells,
            activation_class=nn.ReLU,
            dropout=dropout,
        )

        # Empty sets are represented by a learnable vector
        self.empty_set_vector = nn.Parameter(torch.zeros(output_size))

    @override
    def forward(self, input_set: Tensor, lengths: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input_set: a tensor of shape (batch_size, set_size, element_size). The valid elements have to be first in the sequence.
            lengths: a tensor of shape (batch_size). Number of valid elements in each sequence.

        Returns:
            output: a tensor of shape (batch_size, output_size)

        """
        # We want to support empty sets as well, but these have to be handled separately, look at the end of the function
        original_batch_size = input_set.shape[0]
        nonempty_set_mask = lengths > 0

        # Now only treat the non-empty sets
        input_set = input_set[nonempty_set_mask]
        lengths = lengths[nonempty_set_mask]
        batch_size, set_size, _ = input_set.shape

        # Read: Map each set elements to a memory vector
        memories = self.reading_block(input_set)  # (batch_size, set_size, memory_size)

        # Initialize lstm state
        q_t = torch.zeros(batch_size, self.memory_size, device=input_set.device)
        c_t = torch.zeros(batch_size, self.memory_size, device=input_set.device)
        # Initial input
        r_t = torch.zeros(batch_size, self.memory_size, device=input_set.device)

        # Process: Iteratively refine state with attention over memories
        for _ in range(self.processing_steps):
            # LSTM update
            q_t, c_t = self.process_lstm(
                r_t, (q_t, c_t)
            )  # q_t: (batch_size, memory_size)

            # Attention between each memory vector and all memories, even the invalid ones, to produce logits
            e_t = torch.bmm(memories, q_t.unsqueeze(-1)).squeeze(
                -1
            )  # (batch_size, set_size)

            # Mask padding elements
            mask = torch.arange(set_size, device=input_set.device).repeat(
                batch_size, 1
            ) < lengths.unsqueeze(1).repeat(1, set_size)
            e_t[~mask] = float("-inf")

            # Compute attention weights
            a_t = torch.softmax(e_t, dim=-1)  # (batch_size, set_size)

            # Read from memory with attention
            r_t = torch.bmm(a_t.unsqueeze(1), memories).squeeze(
                1
            )  # (batch_size, memory_size)

        # Write: Transform final state to output
        output = self.write_block(
            torch.cat((q_t, r_t), dim=1)
        )  # (batch_size, output_size)

        # Empty sets are represented by a learnable vector
        complete_output = self.empty_set_vector.expand(original_batch_size, -1).clone()
        complete_output[nonempty_set_mask] = output

        return complete_output


class CopiedMLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_sizes,
        output_size,
        bias=True,
        dropout=False,
        p=0,
        group_norm=0,
        batch_norm=False,
    ):
        super().__init__()
        self.layers = []
        self.n_features = int(input_size / 2)
        in_size = input_size
        cnt = 0
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(in_size, hidden_size, bias=bias))
            if group_norm > 0 and cnt == 0:
                cnt += 1
                self.w0 = self.layers[-1].weight
                print(self.w0.size())
                assert self.w0.size()[1] == input_size
            if batch_norm:
                print("Batchnorm")
                self.layers.append(nn.BatchNorm1d(hidden_size))
            self.layers.append(nn.ReLU())
            if dropout:  # for classifier
                print("Dropout!")
                assert p > 0 and p < 1
                self.layers.append(nn.Dropout(p=p))
            in_size = hidden_size
        self.layers.append(nn.Linear(in_size, output_size, bias=bias))
        if batch_norm:  # FIXME is it good?
            print("Batchnorm")
            self.layers.append(nn.BatchNorm1d(output_size))
        self.layers = nn.ModuleList(self.layers)

        self.output_size = output_size

    def forward(self, x, length=None):
        for layer in self.layers:
            x = layer(x)
        return x


class CopiedSetEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        n_features,
        embedder_hidden_sizes,
        embedded_dim,
        lstm_size,
        n_shuffle,
        simple=True,
        proj_dim=None,
        normalize=False,
        dropout=False,
        p=0,
    ):
        # embedder + lstm
        super().__init__()

        self.n_shuffle = n_shuffle
        self.embedder = CopiedMLP(
            input_dim, embedder_hidden_sizes, embedded_dim, dropout=dropout, p=p
        )
        self.lstm = nn.LSTMCell(embedded_dim, lstm_size)
        # self.module_list = nn.ModuleList([self.embedder, self.lstm])
        self.n_features = n_features
        self.normalize = normalize

        self.lstm_size = lstm_size
        self.embedded_dim = embedded_dim

        if not simple:
            assert proj_dim is not None
            self.attention = nn.ModuleList(
                [
                    nn.Linear(lstm_size, proj_dim, bias=False),
                    nn.Linear(embedded_dim, proj_dim, bias=True),
                    nn.Linear(proj_dim, 1, bias=True),
                ]
            )
        elif embedded_dim != lstm_size:
            self.attention = torch.nn.Linear(lstm_size, embedded_dim, bias=False)
            # torch.nn.init.xavier_normal(self.attention.weight)
            # module.apply(weight_xavier_init)

        def _compute_attention_sum(q, m, length):
            # q : batch_size x lstm_size
            # m : batch_size x max(length) x embedded_dim
            assert torch.max(length) == m.size()[1]
            max_len = m.size()[1]
            if simple:
                if q.size()[-1] != m.size()[-1]:
                    q = self.attention(q)  # batch_size x embedded_dim
                weight_logit = torch.bmm(m, q.unsqueeze(-1)).squeeze(
                    2
                )  # batch_size x n_features
            else:
                linear_m = self.attention[1]
                linear_q = self.attention[0]
                linear_out = self.attention[2]

                packed = pack(m, list(length), batch_first=True)
                proj_m = PackedSequence(linear_m(packed.data), packed.batch_sizes)
                proj_m, _ = pad(
                    proj_m, batch_first=True
                )  # batch_size x n_features x proj_dim
                proj_q = linear_q(q).unsqueeze(1)  # batch_size x 1 x proj_dim
                packed = pack(F.relu(proj_m + proj_q), list(length), batch_first=True)
                weight_logit = PackedSequence(
                    linear_out(packed.data), packed.batch_sizes
                )
                weight_logit, _ = pad(
                    weight_logit, batch_first=True
                )  # batch_size x n_features x 1
                weight_logit = weight_logit.squeeze(2)

            # max_len = weight_logit.size()[1]
            indices = torch.arange(
                0, max_len, out=torch.LongTensor(max_len).unsqueeze(0)
            ).cuda()
            # TODO here.. cuda..
            mask = indices < length.unsqueeze(1)  # .long()
            weight_logit[~mask] = -np.inf
            weight = F.softmax(weight_logit, dim=1)  # nonzero x max_len
            weighted = torch.bmm(weight.unsqueeze(1), m)
            # batch_size x 1 x max_len
            # batch_size x     max_len x embedded_dim
            # = batch_size x 1 x embedded_dim
            return weighted.squeeze(1), weight  # nonzero x embedded_dim

        self.attending = _compute_attention_sum

    def forward(self, state, length):
        """
        Args:
            state: a tensor of shape (batch_size, n_features, input_dim)
            length: a tensor of shape (batch_size) containing the length of each sequence

        Returns:
            encoded: a tensor of shape (batch_size, embedded_dim)
        """
        # length should be sorted
        assert len(state.size()) == 3  # batch x n_features x input_dim
        # input_dim == n_features + 1
        batch_size = state.size()[0]
        self.weight = np.zeros(
            (int(batch_size), self.n_features)
        )  # state.data.new(int(batch_size), self.n_features).fill_(0.)
        nonzero = torch.sum(length > 0).cpu().numpy()  # encode only nonzero points
        if nonzero == 0:
            return state.new(int(batch_size), self.lstm_size + self.embedded_dim).fill_(
                0.0
            )

        length_ = list(length[:nonzero].cpu().numpy())
        packed = pack(state[:nonzero], length_, batch_first=True)

        embedded = self.embedder(packed.data)

        if self.normalize:
            embedded = F.normalize(embedded, dim=1)
        embedded = PackedSequence(embedded, packed.batch_sizes)
        embedded, _ = pad(
            embedded, batch_first=True
        )  # nonzero x max(length) x embedded_dim

        # define initial state
        qt = embedded.new(embedded.size()[0], self.lstm_size).fill_(0.0)
        ct = embedded.new(embedded.size()[0], self.lstm_size).fill_(0.0)

        ###########################
        # shuffling (set encoding)
        ###########################

        for i in range(self.n_shuffle):
            attended, weight = self.attending(qt, embedded, length[:nonzero])
            # attended : nonzero x embedded_dim
            qt, ct = self.lstm(attended, (qt, ct))

        # TODO edit here!
        weight = weight.detach().cpu().numpy()
        tmp = state[:, :, 1:]
        val, acq = torch.max(tmp, 2)  # batch x n_features
        tmp = (val.long() * acq).cpu().numpy()
        # tmp = tmp.cpu().numpy()
        tmp = tmp[: weight.shape[0], : weight.shape[1]]
        self.weight[np.arange(nonzero).reshape(-1, 1), tmp] = weight

        encoded = torch.cat((attended, qt), dim=1)
        if batch_size > nonzero:
            encoded = torch.cat(
                (
                    encoded,
                    encoded.new(int(batch_size - nonzero), encoded.size()[1]).fill_(
                        0.0
                    ),
                ),
                dim=0,
            )
        return encoded


class CopiedShim2018Embedder(Embedder):
    def __init__(self, encoder: CopiedSetEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Embedding:
        feature_set, lengths = get_feature_set(masked_features, feature_mask)

        # Encoder expects set to be order in decreasing lengths
        lengths, indices = torch.sort(lengths, descending=True)
        feature_set = feature_set[indices]

        return self.encoder(feature_set, lengths)


@final
class Shim2018Embedder(Embedder):
    """Wrap a ReadProcessEmbedder to handle inputs consisting of features and their indices, using the representation described in "Joint Active Feature Acquisition and Classification with Variable-Size Set Encoding."""

    def __init__(self, encoder: ReadProcessEncoder):
        super().__init__()
        self.encoder = encoder

    @override
    def forward(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Embedding:
        feature_set, lengths = get_feature_set(
            masked_features, feature_mask
        )  # (batch_size, n_features, state_size)

        # Shuffling set should improve training
        feature_set = shuffle_feature_set(feature_set, lengths)

        return self.encoder(feature_set, lengths)


@final
class Shim2018MLPClassifier(EmbeddingClassifier):
    def __init__(self, input_size: int, num_classes: int, num_cells: tuple[int, ...]):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_cells = num_cells

        self.mlp = MLP(
            input_size, num_classes, num_cells=num_cells, activation_class=nn.ReLU
        )

    @override
    def forward(self, embedding: Embedding) -> Logits:
        return self.mlp(embedding)


@final
class LitShim2018EmbedderClassifier(pl.LightningModule):
    """A module that combines the ShimEncoder with a classifier for pretraining."""

    def __init__(
        self,
        embedder: Shim2018Embedder | CopiedShim2018Embedder,
        classifier: Shim2018MLPClassifier,
        class_probabilities: Float[Tensor, "n_classes"],
        max_masking_probability: float = 1.0,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["embedder", "classifier"])
        self.lr = lr
        self.embedder = embedder
        self.classifier = classifier
        self.class_weight = 1 / class_probabilities
        self.class_weight = self.class_weight / torch.sum(self.class_weight)
        self.max_masking_probability = max_masking_probability

    @override
    def forward(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> tuple[Embedding, Logits]:
        """Forward pass.

        Args:
            masked_features: currently observed features, with zeros for missing features
            feature_mask: indicator for missing features, 1 if feature is observed, 0 if missing

        Returns:
            embedding: the embedding of the input features
            classifier_output: the output of the classifier

        """
        embedding = self.embedder(masked_features, feature_mask)
        classifier_output = self.classifier(embedding)
        return embedding, classifier_output

    @override
    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int):
        features: Features = batch[0]
        label: Label = batch[1]

        masking_probability = torch.rand(1).item() * self.max_masking_probability
        self.log("masking_probability", masking_probability, sync_dist=True)

        masked_features, feature_mask, _ = mask_data(features, p=masking_probability)
        _, y_hat = self(masked_features, feature_mask)
        loss = F.cross_entropy(y_hat, label, weight=self.class_weight.to(y_hat.device))
        self.log("train_loss", loss)
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

    @override
    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int):
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

    @override
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


@final
class Shim2018AFAPredictFn(AFAPredictFn):
    """A wrapper for the ShimEmbedderClassifier to make it compatible with the AFAPredictFn interface."""

    def __init__(self, embedder_and_classifier: LitShim2018EmbedderClassifier):
        super().__init__()
        self.embedder_and_classifier = embedder_and_classifier

    @override
    def __call__(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Label:
        _, logits = self.embedder_and_classifier(masked_features, feature_mask)
        return logits.softmax(dim=-1)


@final
class Shim2018AFAClassifier(AFAClassifier):
    """A wrapper for the ShimEmbedderClassifier to make it compatible with the AFAClassifier interface."""

    def __init__(
        self,
        embedder_and_classifier: LitShim2018EmbedderClassifier,
        device: torch.device,
    ):
        super().__init__()
        self._device = device
        self.embedder_and_classifier = embedder_and_classifier.to(self._device)

    @override
    def __call__(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Label:
        original_device = masked_features.device

        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)

        _, logits = self.embedder_and_classifier(masked_features, feature_mask)
        return logits.softmax(dim=-1).to(original_device)

    @override
    def save(self, path: Path) -> None:
        torch.save(self.embedder_and_classifier.cpu(), path)

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        embedder_and_classifier = torch.load(
            path, weights_only=False, map_location=device
        )
        return cls(embedder_and_classifier, device)

    @override
    def to(self, device: torch.device) -> Self:
        self.embedder_and_classifier = self.embedder_and_classifier.to(device)
        return self

    @property
    @override
    def device(self) -> torch.device:
        return self._device


# class Shim2018MaskedClassifier(MaskedClassifier):
#     """A wrapper for the ShimEmbedderClassifier to make it compatible with the MaskedClassifier interface."""
#
#     def __init__(self, embedder_and_classifier: LitShim2018EmbedderClassifier):
#         self.embedder_and_classifier: LitShim2018EmbedderClassifier = (
#             embedder_and_classifier
#         )
#
#     def __call__(
#         self, masked_features: MaskedFeatures, feature_mask: FeatureMask
#     ) -> Logits:
#         model_device = next(self.embedder_and_classifier.parameters()).device
#         masked_features = masked_features.to(model_device)
#         feature_mask = feature_mask.to(model_device)
#         with torch.no_grad():
#             embedding, logits = self.embedder_and_classifier.forward(
#                 masked_features, feature_mask
#             )
#         return logits.cpu()
