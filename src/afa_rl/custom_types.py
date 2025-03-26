from abc import ABC, abstractmethod
from typing import NamedTuple, Protocol

import torch
from jaxtyping import Bool, Float, Shaped
from torch import Tensor, nn

Feature = Float[Tensor, "*batch feature_size"]
FeatureMask = Bool[Tensor, "*batch feature_size"]
TaskLabel = Shaped[Tensor, "*batch label_size"]
TaskModelOutput = Float[Tensor, "batch model_output_size"]
State = Float[
    Tensor, "*batch state_size"
]  # A state is a concatenation of feature values and feature indices
Embedding = Float[Tensor, "*batch embedding_size"]
FeatureSet = Float[
    Tensor, "batch features {feature_size+1}"
]  # A feature set is the set version of State. Each element-index tuple becomes a vector.


class Sample(NamedTuple):
    feature: Feature
    label: TaskLabel


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


class TaskModel(nn.Module, ABC):
    """
    A task model is typically a nn.Module that takes in an embedding and returns a prediction.
    """

    @abstractmethod
    def forward(self, embedding: Embedding) -> TaskModelOutput: ...

    def __call__(self, embedding: Embedding) -> TaskModelOutput:
        return super().__call__(embedding)
