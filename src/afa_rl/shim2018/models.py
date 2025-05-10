from typing import Tuple

import lightning as pl
from pandas.core.arrays import masked
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
    shuffle_feature_set,
)
from common.custom_types import FeatureMask, MaskedFeatures
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pad_packed_sequence as pad


class ReadProcessEncoder(nn.Module):
    """A set encoder using RNNs, as described in the paper "Order Matters: Sequence to sequence for sets" (http://arxiv.org/abs/1511.06391)."""

    def __init__(
        self,
        set_element_size: int,  # size of each element in input set. Usually 1 (scalar sequence)
        output_size: int,  # size of output vector, one output per sequence
        reading_block_cells: tuple[int,...] = (32, 32),
        writing_block_cells: tuple[int,...] = (32, 32),
        memory_size: int = 16,  # each element in the sequence gets converted into a memory with this size
        processing_steps: int = 5,  # RNN processing steps. Paper shows that 5-10 is good.
    ):
        super().__init__()
        self.set_element_size = set_element_size
        self.output_size = output_size
        self.reading_block_cells = reading_block_cells
        self.writing_block_cells = writing_block_cells
        self.memory_size = memory_size
        self.processing_steps = processing_steps

        self.reading_block = MLP(
            in_features=set_element_size,
            out_features=memory_size,
            num_cells=reading_block_cells,
            activation_class=nn.ReLU,
        )
        # An RNN without input! The input will always be zero and the sequence length will be 1
        self.process_lstm = nn.LSTMCell(1, 2*memory_size)

        # The RNN output has to be projected to the memory size
        self.proj = nn.Linear(2 * memory_size, memory_size)

        # After the processing steps, the final memory is passed through a MLP to produce the output
        # The paper uses a pointer network but we only want a single feature vector
        self.write_block = MLP(
            in_features=2 * memory_size,
            out_features=output_size,
            num_cells=writing_block_cells,
            activation_class=nn.ReLU,
        )

        # Empty sets are represented by a learnable vector
        self.empty_set_vector = nn.Parameter(torch.zeros(output_size))
        # torch.nn.init.xavier_normal_(self.empty_set_vector)


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
        batch_size, set_size, element_size = input_set.shape

        # Read: Map each set elements to a memory vector
        memories = self.reading_block(input_set)  # (batch_size, set_size, memory_size)

        # Initialize Process block state
        q_star_t = torch.zeros(batch_size, 2*self.memory_size, device=input_set.device)
        c_t = torch.zeros(batch_size, 2*self.memory_size, device=input_set.device)

        # Process: Iteratively refine state with attention over memories
        for _ in range(self.processing_steps):
            # LSTM update
            q_t, c_t = self.process_lstm(torch.zeros(batch_size, 1, device=input_set.device), (q_star_t, c_t)) # q_t: (batch_size, 2*memory_size)

            # Project LSTM output to memory size
            q_t = self.proj(q_t)  # (batch_size, memory_size)

            # Attention between each memory vector and all memories, even the invalid ones, to produce logits
            e_t = torch.bmm(memories, q_t.unsqueeze(-1)).squeeze(-1)  # (batch_size, set_size)

            # Mask padding elements
            mask = torch.arange(set_size, device=input_set.device).repeat(batch_size, 1) < lengths.unsqueeze(1).repeat(1, set_size)
            e_t[~mask] = float('-inf')

            # Compute attention weights
            a_t = torch.softmax(e_t, dim=-1)  # (batch_size, set_size)

            # Read from memory with attention
            r_t = torch.bmm(a_t.unsqueeze(1), memories).squeeze(1)  # (batch_size, memory_size)

            # Concatenate query and read vectors
            q_star_t = torch.cat([q_t, r_t], dim=-1)  # (batch_size, 2*memory_size)

        # Write: Transform final state to output
        output = self.write_block(q_star_t)

        # Empty sets are represented by a learnable vector
        complete_output = torch.zeros(original_batch_size, self.output_size, device=input_set.device)
        complete_output[nonempty_set_mask] = output
        complete_output[~nonempty_set_mask] = self.empty_set_vector

        return complete_output

class CopiedMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, bias=True,
            dropout=False, p=0, group_norm=0, batch_norm=False):
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
            if dropout: # for classifier
                print("Dropout!")
                assert p > 0 and p < 1
                self.layers.append(nn.Dropout(p=p))
            in_size = hidden_size
        self.layers.append(nn.Linear(in_size, output_size, bias=bias))
        if batch_norm: # FIXME is it good?
            print("Batchnorm")
            self.layers.append(nn.BatchNorm1d(output_size))
        self.layers = nn.ModuleList(self.layers)

        self.output_size = output_size

    def forward(self, x, length=None):
        for layer in self.layers:
            x = layer(x)
        return x

class CopiedSetEncoder(nn.Module):
    def __init__(self,
                 input_dim, n_features,
                 embedder_hidden_sizes, embedded_dim,
                 lstm_size, n_shuffle,
                 simple=True, proj_dim=None, normalize=False,
                 dropout=False, p=0):
        # embedder + lstm
        super().__init__()

        self.n_shuffle = n_shuffle
        self.embedder = CopiedMLP(input_dim, embedder_hidden_sizes, embedded_dim,
                dropout=dropout, p=p)
        self.lstm = nn.LSTMCell(embedded_dim, lstm_size)
        #self.module_list = nn.ModuleList([self.embedder, self.lstm])
        self.n_features = n_features
        self.normalize = normalize

        self.lstm_size = lstm_size
        self.embedded_dim = embedded_dim

        if not simple:
            assert proj_dim is not None
            self.attention = nn.ModuleList(
                [nn.Linear(lstm_size, proj_dim, bias=False),
                 nn.Linear(embedded_dim, proj_dim, bias=True),
                 nn.Linear(proj_dim, 1, bias=True)]
            )
        elif embedded_dim != lstm_size:
            self.attention = torch.nn.Linear(lstm_size, embedded_dim,
                    bias=False)
            # torch.nn.init.xavier_normal(self.attention.weight)
            # module.apply(weight_xavier_init)

        def _compute_attention_sum(q, m, length):
            # q : batch_size x lstm_size
            # m : batch_size x max(length) x embedded_dim
            assert torch.max(length) == m.size()[1]
            max_len = m.size()[1]
            if simple:
                if q.size()[-1] != m.size()[-1]:
                    q = self.attention(q) # batch_size x embedded_dim
                weight_logit = torch.bmm(m, q.unsqueeze(-1)).squeeze(2) # batch_size x n_features
            else:
                linear_m = self.attention[1]
                linear_q = self.attention[0]
                linear_out = self.attention[2]

                packed = pack(m, list(length), batch_first=True)
                proj_m = PackedSequence(linear_m(packed.data), packed.batch_sizes)
                proj_m, _ = pad(proj_m, batch_first=True)  # batch_size x n_features x proj_dim
                proj_q = linear_q(q).unsqueeze(1) # batch_size x 1 x proj_dim
                packed = pack(F.relu(proj_m + proj_q), list(length), batch_first=True)
                weight_logit = PackedSequence(linear_out(packed.data), packed.batch_sizes)
                weight_logit, _ = pad(weight_logit, batch_first=True) # batch_size x n_features x 1
                weight_logit = weight_logit.squeeze(2)

            # max_len = weight_logit.size()[1]
            indices = torch.arange(0, max_len,
                out=torch.LongTensor(max_len).unsqueeze(0)).cuda()
            # TODO here.. cuda..
            mask = indices < length.unsqueeze(1)#.long()
            weight_logit[~mask] = -np.inf
            weight = F.softmax(weight_logit, dim=1) # nonzero x max_len
            weighted = torch.bmm(weight.unsqueeze(1), m)
            # batch_size x 1 x max_len
            # batch_size x     max_len x embedded_dim
            # = batch_size x 1 x embedded_dim
            return weighted.squeeze(1), weight  #nonzero x embedded_dim

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
        assert len(state.size()) == 3 # batch x n_features x input_dim
                                      # input_dim == n_features + 1
        batch_size = state.size()[0]
        self.weight = np.zeros((int(batch_size), self.n_features))#state.data.new(int(batch_size), self.n_features).fill_(0.)
        nonzero = torch.sum(length > 0).cpu().numpy() # encode only nonzero points
        if nonzero == 0:
            return state.new(int(batch_size), self.lstm_size + self.embedded_dim).fill_(0.)

        length_ = list(length[:nonzero].cpu().numpy())
        packed = pack(state[:nonzero], length_, batch_first=True)

        embedded = self.embedder(packed.data)


        if self.normalize:
            embedded = F.normalize(embedded, dim=1)
        embedded = PackedSequence(embedded, packed.batch_sizes)
        embedded, _ = pad(embedded, batch_first=True) # nonzero x max(length) x embedded_dim

        # define initial state
        qt = embedded.new(embedded.size()[0], self.lstm_size).fill_(0.)
        ct = embedded.new(embedded.size()[0], self.lstm_size).fill_(0.)

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
        val, acq = torch.max(tmp, 2) # batch x n_features
        tmp = (val.long() * acq).cpu().numpy()
        #tmp = tmp.cpu().numpy()
        tmp = tmp[:weight.shape[0], :weight.shape[1]]
        self.weight[np.arange(nonzero).reshape(-1, 1), tmp] = weight

        encoded = torch.cat((attended, qt), dim=1)
        if batch_size > nonzero:
            encoded = torch.cat(
                (encoded,
                 encoded.new(int(batch_size - nonzero),
                     encoded.size()[1]).fill_(0.)),
                dim=0
            )
        return encoded

class CopiedShim2018Embedder(Embedder):
    def __init__(self, encoder: CopiedSetEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Embedding:
        feature_set = get_feature_set(masked_features, feature_mask)
        batch_size = masked_features.shape[0]
        n_features = feature_set.shape[1]
        length = torch.full((batch_size,), n_features, dtype=torch.long, device=masked_features.device)
        return self.encoder(feature_set, length)


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
        feature_set, lengths = get_feature_set(masked_features, feature_mask) # (batch_size, n_features, state_size)

        # Shuffling set should improve training
        feature_set = shuffle_feature_set(feature_set, lengths)

        return self.encoder(feature_set, lengths)


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
        # return loss_optimal
        # masked_features = feature_values_optimal
        # feature_mask = feature_mask_optimal

        masked_features, feature_mask, nonzero_mask = mask_data(features, p=self.masking_probability)
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
        return logits.cpu()
