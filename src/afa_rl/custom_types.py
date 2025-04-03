from abc import ABC, abstractmethod
from typing import NamedTuple, Protocol

import torch
from jaxtyping import Bool, Float, Integer
from torch import Tensor, nn

Feature = Float[Tensor, "*batch feature_size"]
FeatureMask = Bool[Tensor, "*batch feature_size"]
Label = Integer[Tensor, "*batch label_size"]
Logits = Float[Tensor, "batch model_output_size"]
State = Float[
    Tensor, "*batch state_size"
]  # A state is a concatenation of feature values and feature indices
Embedding = Float[Tensor, "*batch embedding_size"]
FeatureSet = Float[
    Tensor, "batch features {feature_size+1}"
]  # A feature set is the set version of State. Each element-index tuple becomes a vector.


class Sample(NamedTuple):
    feature: Feature
    label: Label


class DatasetFn(Protocol):
    """
    A dataset that returns batched samples. move_on determines whether the dataset should return different samples next time the function is called.
    """

    def __call__(self, batch_size: torch.Size, move_on: bool = True) -> Sample: ...


class Embedder(nn.Module, ABC):
    """
    An Embedder converts feature values and feature indices (1 if a feature is observed, 0 if not) to an embedding.
    """

    @abstractmethod
    def forward(
        self, feature_values: Feature, feature_mask: FeatureMask
    ) -> Embedding: ...

    def __call__(self, feature_values: Feature, feature_mask: FeatureMask) -> Embedding:
        return super().__call__(feature_values, feature_mask)


class Classifier(nn.Module, ABC):
    @abstractmethod
    def forward(self, embedding: Embedding) -> Logits: ...

    def __call__(self, embedding: Embedding) -> Logits:
        return super().__call__(embedding)
