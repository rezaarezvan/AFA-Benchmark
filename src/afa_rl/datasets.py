import math
from typing import final

import lightning as pl
import torch
from jaxtyping import Shaped
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from afa_rl.custom_types import AFADatasetFn
from common.custom_types import FeatureMask, Features, Label, MaskedFeatures


def get_wrapped_batch(
    t: Shaped[Tensor, "batch *rem"], idx: int, numel: int
) -> Shaped[Tensor, "{num_elems} *rem"]:
    """Get a batch of size num_elems from a tensor t, starting at index idx, wrapping around if necessary."""
    n = len(t)
    repeated = t.repeat((numel // n) + 2, *[1] * (t.ndim - 1))
    return repeated[idx : idx + numel]


def get_afa_dataset_fn(
    features: Features, labels: Label, device: torch.device | None = None
) -> AFADatasetFn:
    """Given features and labels, return a function that can be used to get batches of AFA data."""
    idx = 0  # keep track of where in the dataset we are

    def afa_dataset_fn(
        batch_size: torch.Size,
        move_on: bool = True,  # noqa: FBT002
    ) -> tuple[Features, Label]:
        nonlocal idx, features, labels
        local_features = get_wrapped_batch(features, idx, batch_size.numel())
        local_labels = get_wrapped_batch(labels, idx, batch_size.numel())
        if move_on:
            idx = idx + batch_size.numel()
            # Reset idx if needed, also shuffling the dataset
            if idx >= len(features):
                idx = 0
                # Shuffle the dataset
                perm = torch.randperm(len(features))
                features = features[perm]
                labels = labels[perm]
        local_features = local_features.reshape(
            *batch_size, local_features.shape[-1]
        )
        local_labels = local_labels.reshape(
            *batch_size, local_labels.shape[-1]
        )

        # Move to specified device if provided
        if device is not None:
            local_features = local_features.to(device)
            local_labels = local_labels.to(device)

        return local_features, local_labels

    return afa_dataset_fn


@final
class DataModuleFromDatasets(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        batch_size=32,
        num_workers=0,
        persistent_workers=False,
    ):
        # TODO: does not work with num_workers > 1
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )


class OneHotLabelWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, num_classes):
        self.dataset = dataset
        self.num_classes = num_classes

    def __getitem__(self, index):
        img, label = self.dataset[index]
        one_hot_label = F.one_hot(
            torch.tensor(label), num_classes=self.num_classes
        )
        return img, one_hot_label

    def __len__(self):
        return len(self.dataset)


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=128, transform=None):
        super().__init__()
        self.batch_size = batch_size
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    def prepare_data(self):
        datasets.MNIST(root="data/MNIST", train=True, download=True)
        datasets.MNIST(root="data/MNIST", train=False, download=True)

    def setup(self, stage=None):
        mnist_full = datasets.MNIST(
            root="data/MNIST", train=True, transform=self.transform
        )
        train_set, val_set = random_split(mnist_full, [55000, 5000])
        self.train_set = OneHotLabelWrapper(train_set, num_classes=10)
        self.val_set = OneHotLabelWrapper(val_set, num_classes=10)

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)


class Zannone2019CubeDataset(Dataset):
    """
    The Cube dataset, as described in the paper "ODIN: Optimal Discovery of High-value INformation Using Model-based Deep Reinforcement Learning"

    Implements the AFADataset protocol.
    """

    def __init__(
        self,
        n_features: int = 20,
        data_points: int = 20000,
        seed: int = 123,
        non_informative_feature_mean: float = 0.5,
        informative_feature_variance: float = 0.1,
        non_informative_feature_variance: float = 0.3,
    ):
        super().__init__()
        self.n_features = n_features
        self.data_points = data_points
        self.seed = seed
        self.non_informative_feature_mean = non_informative_feature_mean
        self.informative_feature_variance = informative_feature_variance
        self.non_informative_feature_variance = (
            non_informative_feature_variance
        )

        self._informative_feature_std = math.sqrt(informative_feature_variance)
        self._non_informative_feature_std = math.sqrt(
            self.non_informative_feature_variance
        )

    def generate_data(self) -> None:
        rng = torch.Generator()
        rng.manual_seed(self.seed)
        # Each coordinate is drawn from a Bernoulli distribution with p=0.5, which is the same as uniform
        coords = torch.randint(
            low=0,
            high=2,
            size=(self.data_points, 3),
            dtype=torch.int64,
            generator=rng,
        )
        # Each corner in the cube is a different label
        labels = torch.einsum(
            "bi,i->b", coords, torch.tensor([1, 2, 4], dtype=torch.int64)
        )
        # Coords have noise
        coords = coords.float()
        coords += (
            torch.randn(
                self.data_points, 3, dtype=torch.float32, generator=rng
            )
            * self._informative_feature_std
        )
        # The final features are the coordinates offset according to the labels, and some noise added
        self.features = torch.zeros(
            self.data_points, self.n_features, dtype=torch.float32
        )
        for i in range(self.data_points):
            offset: int = labels[i].item()
            self.features[i, offset : offset + 3] += coords[i]
            # All other features have mean 0.5 and variance 0.3
            self.features[i, :offset] = torch.normal(
                mean=self.non_informative_feature_mean,
                std=self._non_informative_feature_std,
                size=(1, offset),
                dtype=torch.float32,
                generator=rng,
            )
            self.features[i, offset + 3 :] = torch.normal(
                mean=self.non_informative_feature_mean,
                std=self._non_informative_feature_std,
                size=(1, self.n_features - offset - 3),
                dtype=torch.float32,
                generator=rng,
            )
        # Convert labels to one-hot encoding
        self.labels = torch.nn.functional.one_hot(labels, num_classes=8)

    def __getitem__(self, idx: int) -> tuple[MaskedFeatures, FeatureMask]:
        return self.features[idx], self.labels[idx]

    def __len__(self):
        return len(self.features)

    def get_all_data(self) -> tuple[MaskedFeatures, FeatureMask]:
        return self.features, self.labels

    def save(self, path: str) -> None:
        torch.save(
            {
                "features": self.features,
                "labels": self.labels,
                "config": {
                    "n_features": self.n_features,
                    "data_points": self.data_points,
                    "seed": self.seed,
                },
            },
            path,
        )

    @staticmethod
    def load(path: str) -> "Zannone2019CubeDataset":
        data = torch.load(path)
        dataset = Zannone2019CubeDataset(**data["config"])
        dataset.features = data["features"]
        dataset.labels = data["labels"]
        return dataset
