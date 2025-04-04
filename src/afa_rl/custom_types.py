from abc import ABC, abstractmethod
from typing import Protocol

import torch
from jaxtyping import Float
from torch import Tensor, nn

from common.custom_types import FeatureMask, Features, Label, MaskedFeatures

Logits = Float[Tensor, "batch model_output_size"]
State = Float[
    Tensor, "*batch state_size"
]  # A state is a concatenation of feature values and feature indices
Embedding = Float[Tensor, "*batch embedding_size"]
FeatureSet = Float[
    Tensor, "batch features {feature_size+1}"
]  # A feature set is the set version of State. Each element-index tuple becomes a vector.


class Embedder(nn.Module, ABC):
    """
    An Embedder converts feature values and feature indices (1 if a feature is observed, 0 if not) to an embedding.
    """

    @abstractmethod
    def forward(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Embedding: ...

    def __call__(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Embedding:
        return super().__call__(masked_features, feature_mask)


class Classifier(nn.Module, ABC):
    @abstractmethod
    def forward(self, embedding: Embedding) -> Logits: ...

    def __call__(self, embedding: Embedding) -> Logits:
        return super().__call__(embedding)


class AFADatasetFn(Protocol):
    """
    A dataset that returns new batched samples in the same format as AFADataset. move_on determines whether the dataset should return different samples next time the function is called.
    """

    def __call__(
        self, batch_size: torch.Size, move_on: bool = True
    ) -> tuple[Features, Label]: ...
