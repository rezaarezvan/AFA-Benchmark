from typing import Protocol

from jaxtyping import Bool, Float, Integer
from torch import Tensor

# AFA datasets return features and labels
type Features = Float[Tensor, "*batch n_features"]
type Label = Bool[Tensor, "*batch n_classes"]


class AFADataset(Protocol):
    """
    Datasets that can be used for evaluating AFA methods.

    Notably, the __init__ constructor should *not* generate data. Instead, generate_data() should be called. This makes it possible to call load if deterministic data is desired.
    """

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

    def save(self, path: str) -> None:
        """
        Saves the dataset to a file. The file should be in a format that can be loaded by the dataset.
        This enables deterministic loading of datasets.
        """
        ...

    @staticmethod
    def load(path: str) -> "AFADataset":
        """
        Loads the dataset from a file. The file should contain the dataset in a format that can be loaded by the dataset.
        This enables deterministic loading of datasets.
        """
        ...


type MaskedFeatures = Integer[Tensor, "*batch n_features"]
type FeatureMask = Bool[Tensor, "*batch n_features"]
type AFASelection = Integer[Tensor, "*batch 1"]


class AFAMethod(Protocol):
    """
    An AFA method is an object that can be called with
    - a tensor of features, 0 for unobserved features
    - a tensor of feature indices, 0 for unobserved features and 1 for observed features
    and returns either
    - 0, meaning that features should stop being collected
    - i > 0, meaning that feature i should be collected

    Furthermore, the `load` method should be called on it after loading it from a pickle.
    """

    def __call__(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
    ) -> AFASelection:
        """
        Returns the index of the feature to be collected next.
        """
        ...

    def save(self, path: str) -> None:
        """
        Saves the method to a file. The file should be in a format that can be loaded by the method.
        """
        ...

    @staticmethod
    def load(path: str) -> "AFAMethod":
        """
        Loads the method from a file.
        """
        ...
