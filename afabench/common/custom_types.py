from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, Self

import torch
from jaxtyping import Bool, Integer
from torch import Tensor

# We need to be able to distinguish between samples, e.g., for tracking performance per sample
type SampleIndex = Integer[Tensor, "*batch 1"]

type Logits = Tensor


class AFADataset(Protocol):
    """
    Datasets that can be used for evaluating AFA methods.

    The constructor should generate data immediately. For deterministic loading,
    use the load() class method which bypasses __init__ using __new__.

    The constructor is expected to accept the following parameters:
        - seed: int | None
    """

    @property
    def feature_shape(self) -> torch.Size:
        """Return the shape of features (excluding batch dimension)."""
        ...

    @property
    def label_shape(self) -> torch.Size:
        """Return the shape of labels (excluding batch dimension)."""
        ...

    @classmethod
    def accepts_seed(cls) -> bool:
        """Return whether the dataset constructor accepts a seed parameter."""
        ...

    def create_subset(self, indices: Sequence[int]) -> Self:
        """
        Return a new dataset instance containing only the specified indices.

        Implementers must provide this method. For in-memory datasets with
        `features` and `labels` attributes, you may inherit from
        `SubsettableMixin` to get a default implementation.
        """
        ...

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Return a single sample from the dataset as (features, label)."""
        ...

    def __len__(self) -> int: ...

    def get_all_data(
        self,
    ) -> tuple[Tensor, Tensor]:
        """Return all of the data in the dataset as (features, labels). Useful for batched computations."""
        ...

    def save(self, path: Path) -> None:
        """Save the dataset to a file or folder. The file/folder should be in a format that can be loaded by the dataset. This enables deterministic loading of datasets."""
        ...

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load the dataset from a file/folder. The file/folder should contain the dataset in a format that can be loaded by the dataset. This enables deterministic loading of datasets."""
        ...

    def get_feature_acquisition_costs(self) -> Tensor:
        """Return the acquisition costs for each feature as a 1D tensor of shape (feature_shape[0],)."""
        return torch.ones(
            self.feature_shape[0]
        )  # Default: all features have cost 1


type MaskedFeatures = Tensor
type FeatureMask = Bool[Tensor, "*batch *feature_shape"]

# Outputs of AFA methods, representing which feature to collect next, or to stop acquiring features (0)
type AFASelection = Integer[Tensor, "*batch 1"]


class AFAMethod(Protocol):
    """An AFA method is an object that can decide which features to collect next (or stop collecting features) and also do predictions with the features it has seen so far."""

    def select(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features: Tensor | None,
        label: Tensor | None,
    ) -> AFASelection:
        """Return the 0-based index of the feature to be collected."""
        ...

    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features: Tensor | None,
        label: Tensor | None,
    ) -> Tensor:
        """Return the predicted label for the features that have been observed so far."""
        ...

    def save(self, path: Path) -> None:
        """Save the method to disk. The folder should be in a format that can be loaded by the method."""
        ...

    @classmethod
    def load(cls, path: Path, device: torch.device) -> Self:
        """Load the method from a folder, placing it on the given device."""
        ...

    def to(self, device: torch.device) -> Self:
        """Move the object to the specified device. This should determine on which device calculations will be performed."""
        ...

    @property
    def device(self) -> torch.device:
        """Return the current device the method is on."""
        ...

    @property
    def has_builtin_classifier(self) -> bool:
        """Return the current device the method is on."""
        return False

    @property
    def cost_param(self) -> float | None:
        """Return the cost parameter, if any. Only applies to methods that make trade-offs between feature cost and accuracy."""
        return None

    def set_cost_param(self, cost_param: float) -> None:
        """Set the cost parameter, if any. Mostly applies to methods that do not need a cost parameter during training but can adjust the trade-off during evaluation."""
        pass  # noqa: PIE790


class AFAClassifier(Protocol):
    """
    An AFA classifier is an object that can perform classification on masked features.

    Classifiers saved as artifacts should follow this protocol to ensure compatibility with the evaluation scripts.
    """

    def __call__(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features: Tensor | None,
        label: Tensor | None,
    ) -> Tensor:
        """Return the predicted label for the features that have been observed so far."""
        ...

    def save(self, path: Path) -> None:
        """Save the classifier to a file or folder. The file/folder should be in a format that can be loaded by the method."""
        ...

    @classmethod
    def load(cls, path: Path, device: torch.device) -> Self:
        """Load the classifier from a file or folder, placing it on the given device."""
        ...

    def to(self, device: torch.device) -> Self:
        """Move the object to the specified device. This should determine on which device calculations will be performed."""
        ...

    @property
    def device(self) -> torch.device:
        """Return the current device the method is on."""
        ...


# Feature selection interface assumed during evaluation
class AFASelectFn(Protocol):
    def __call__(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features: Tensor | None,
        label: Tensor | None,
    ) -> AFASelection: ...


# Classifier prediction interface assumed during evaluation
class AFAPredictFn(Protocol):
    def __call__(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features: Tensor | None,
        label: Tensor | None,
    ) -> Tensor: ...


# Feature uncover interface assumed during evaluation. Applicable in scenarios where a single AFA action might uncover multiple features.
class AFAUncoverFn(Protocol):
    def __call__(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features: Tensor,
        afa_selection: AFASelection,
    ) -> tuple[FeatureMask, MaskedFeatures]: ...
