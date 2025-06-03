from pathlib import Path
from typing import ClassVar, Protocol, Self
from collections.abc import Callable

from jaxtyping import Bool, Float, Integer
from torch import Tensor
import torch

# AFA datasets return features and labels
type Features = Float[Tensor, "*batch n_features"]
# We use float here since in general we can have probabilities, not only one-hot
type Label = Float[Tensor, "*batch n_classes"]

type Logits = Float[Tensor, "*batch model_output_size"]


class AFADataset(Protocol):
    """Datasets that can be used for evaluating AFA methods.

    Notably, the __init__ constructor should *not* generate data. Instead, generate_data() should be called. This makes it possible to call load if deterministic data is desired.
    """

    # Used by AFADatasetFn
    features: Features  # batched
    labels: Label  # batched

    # Used by evaluation scripts to avoid loading the dataset
    n_classes: ClassVar[int]
    n_features: ClassVar[int]

    def generate_data(self) -> None:
        """Previously generated data but now does nothing. Purely for backwards compatibility."""
        ...

    def __getitem__(self, idx: int) -> tuple[Features, Label]:
        """Return a single (possibly batched) sample from the dataset."""
        ...

    def __len__(self) -> int: ...

    def get_all_data(self) -> tuple[Features, Label]:
        """Return all of the data in the dataset. Useful for batched computations."""
        ...

    def save(self, path: Path) -> None:
        """Save the dataset to a file. The file should be in a format that can be loaded by the dataset. This enables deterministic loading of datasets."""
        ...

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load the dataset from a file. The file should contain the dataset in a format that can be loaded by the dataset. This enables deterministic loading of datasets."""
        ...


type MaskedFeatures = Integer[Tensor, "*batch n_features"]
type FeatureMask = Bool[Tensor, "*batch n_features"]

# Outputs of AFA methods, representing which feature to collect next.
# It is not possible to choose to stop collecting features, since we use a hard budget.
type AFASelection = Integer[Tensor, "*batch 1"]


class AFAMethod(Protocol):
    """An AFA method is an object that can decide which features to collect next and also do predictions with the features it has seen so far.

    The `load` method should be called on it after loading it from a pickle.
    """

    def select(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
    ) -> AFASelection:
        """Return the 1-based index of the feature to be collected next or 0 if no more features should be collected.

        The output tensor should be placed on the same device as the input tensors.
        """
        ...

    def predict(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Label:
        """Return the predicted label for the features that have been observed so far.

        The output tensor should be placed on the same device as the input tensors.
        """
        ...

    def save(self, path: Path) -> None:
        """Save the method to a file or folder. The file or folder should be in a format that can be loaded by the method."""
        ...

    @classmethod
    def load(cls, path: Path, device: torch.device) -> Self:
        """Load the method from a file or folder, placing it on the given device."""
        ...

    def to(self, device: torch.device) -> Self:
        """Move the object to the specified device. This should determine on which device calculations will be performed."""
        ...

    @property
    def device(self) -> torch.device:
        """Return the current device the method is on."""
        ...


class AFAClassifier(Protocol):
    """An AFA classifier is an object that can perform classification on masked features.

    Classifiers saved as artifacts should follow this protocol to ensure compatibility with the evaluation scripts.
    """

    def __call__(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
    ) -> Label:
        """Return the predicted label for the features that have been observed so far.

        The output tensor should be placed on the same device as the input tensors.
        """
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


class AFASelectFn(Protocol):
    def __call__(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> AFASelection: ...


# What we assume during evaluation. Includes both AFAClassifiers and AFAMethod.predict
class AFAPredictFn(Protocol):
    def __call__(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Label: ...


# class PretrainingFunction(Protocol):
#     def __call__(
#         self,
#         pretrain_config_path: Path,
#         dataset_type: str,
#         train_dataset_path: Path,
#         val_dataset_path: Path,
#         pretrained_model_path: Path,
#         seed: int,
#     ) -> None: ...
#
#     """
#     Args:
#         pretrain_config_path: Path to a YAML config file for pretraining. May contain info like learning rate, architecture, etc.
#         dataset_type: The type of dataset to use for pretraining. One of the keys in AFA_DATASET_REGISTRY.
#         train_dataset_path: Path to the .pt file of the training dataset.
#         val_dataset_path: Path to the .pt file of the validation dataset.
#         pretrained_model_path: Path to a folder to save the pretrained model. The model weights will be saved in a model.pt file in this folder, and parameters related to the training will be saved in a params.yml file.
#         seed: Random seed for reproducibility.
#     """
#
#
# class TrainingFunction(Protocol):
#     def __call__(
#         self,
#         pretrain_config_path: Path,
#         train_config_path: Path,
#         dataset_type: str,
#         train_dataset_path: Path,
#         val_dataset_path: Path,
#         pretrained_model_path: Path,
#         hard_budget: int,
#         seed: int,
#         afa_method_path: Path,
#     ) -> None: ...
#
#     """
#         pretrain_config_path: Path to a YAML config file for pretraining. May contain info like learning rate, architecture, etc.
#         train_config_path: Path to a YAML config file for training. May contain info like learning rate, architecture, etc.
#         dataset_type: The type of dataset to use for pretraining. One of the keys in AFA_DATASET_REGISTRY.
#         train_dataset_path: Path to the .pt file of the training dataset.
#         val_dataset_path: Path to the .pt file of the validation dataset.
#         pretrained_model_path: Path to a folder where a pretrained model is saved (see PretrainingFunction).
#         seed: Random seed for reproducibility.
#         afa_method_path: Path to a folder to save the AFA method. The model weights will be saved in a model.pt file in this folder, and parameters related to the training will be saved in a params.yml file.
#     """
