from pathlib import Path
from typing import Callable, ClassVar, Protocol

from jaxtyping import Bool, Float, Integer
from torch import Tensor
import torch

# AFA datasets return features and labels
type Features = Float[Tensor, "*batch n_features"]
# We use float here since in general we can have probabilities, not only one-hot
type Label = Float[Tensor, "*batch n_classes"]

type Logits = Float[Tensor, "*batch model_output_size"]


class AFADataset(Protocol):
    """
    Datasets that can be used for evaluating AFA methods.

    Notably, the __init__ constructor should *not* generate data. Instead, generate_data() should be called. This makes it possible to call load if deterministic data is desired.
    """

    # Used by AFADatasetFn
    features: Features|None # batched
    labels: Label|None # batched

    # Used by evaluation scripts to avoid loading the dataset
    n_classes: ClassVar[int]

    def generate_data(self) -> None:
        """
        Generates the data for the dataset. This should be called after __init__.
        """
        ...

    def __getitem__(self, idx: int) -> tuple[Features, Label]:
        """
        Returns a single (possibly batched) sample from the dataset.
        """
        ...

    def __len__(self) -> int: ...

    def get_all_data(self) -> tuple[Features, Label]:
        """
        Returns all of the data in the dataset. Useful for batched computations.
        """
        ...

    def save(self, path: Path) -> None:
        """
        Saves the dataset to a file. The file should be in a format that can be loaded by the dataset.
        This enables deterministic loading of datasets.
        """
        ...

    @staticmethod
    def load(path: Path) -> "AFADataset":
        """
        Loads the dataset from a file. The file should contain the dataset in a format that can be loaded by the dataset.
        This enables deterministic loading of datasets.
        """
        ...


type MaskedFeatures = Integer[Tensor, "*batch n_features"]
type FeatureMask = Bool[Tensor, "*batch n_features"]

# Outputs of AFA methods, representing which feature to collect next.
# It is not possible to choose to stop collecting features, since we use a hard budget.
type AFASelection = Integer[Tensor, "*batch 1"]


class AFAMethod(Protocol):
    """
    An AFA method is an object that can decide which features to collect next and also do predictions with the features it has seen so far.

    The `load` method should be called on it after loading it from a pickle.
    """

    def select(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
    ) -> AFASelection:
        """
        Returns the 1-based index of the feature to be collected next or 0 if no more features should be collected.
        """
        ...

    def predict(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Label:
        """
        Returns the predicted label for the features that have been observed so far.
        """
        ...

    def save(self, path: Path) -> None:
        """
        Saves the method to a file. The file should be in a format that can be loaded by the method.
        """
        ...

    @staticmethod
    def load(path: Path, device: torch.device) -> "AFAMethod":
        """
        Loads the method from a file, placing it on the given device.
        """
        ...


# A reward function will in general depend on
# - masked features at time t
# - feature mask (the features that have been observed so far) at time t
# - masked features at time x_{t+1}
# - feature mask (the features that have been observed so far) at time t+1
# - the selection (the feature that was selected) at time t
# - the ground truth features
# - the ground truth label

type AFAReward = Float[Tensor, "*batch 1"]
type AFARewardFn = Callable[
    [
        MaskedFeatures,
        FeatureMask,
        MaskedFeatures,
        FeatureMask,
        AFASelection,
        Features,
        Label,
    ],
    AFAReward,
]

class AFAClassifier(Protocol):
    """
    An AFA classifier is an object that can perform classification on masked features.

    """

    def __call__(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
    ) -> Logits:
        """
        Returns the predicted logits for the features that have been observed so far.
        """
        ...

    def save(self, path: Path) -> None:
        """
        Saves the classifier to a file. The file should be in a format that can be loaded by the method.
        """
        ...

    @staticmethod
    def load(path: Path, device: torch.device) -> "AFAClassifier":
        """
        Loads the classifier from a file, placing it on the given device.
        """
        ...
