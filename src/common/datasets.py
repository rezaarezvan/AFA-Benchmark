from pathlib import Path
from typing import Callable, Self, final, override
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import pandas as pd
import os
from torch.nn import functional as F


from common.custom_types import AFADataset, FeatureMask, MaskedFeatures, Features, Label


@final
class Shim2018CubeDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    """
    The Cube dataset, as described in the paper "Minimizing data consumption with sequential online feature selection" (https://doi.org/10.1007/s13042-012-0092-x).
    """

    n_classes = 8
    n_features = 20  # Fixed number of features

    def __init__(
        self,
        n_samples: int = 20000,
        sigma: float = 0.1,
        seed: int = 123,
    ):
        self.n_samples = n_samples
        self.sigma = sigma
        self.seed = seed
        self.rng = torch.Generator()

    @override
    def generate_data(self):
        self.rng.manual_seed(self.seed)

        # Each coordinate is drawn from a Bernoulli distribution with p=0.5, which is the same as uniform
        coords = torch.randint(
            low=0,
            high=2,
            size=(self.n_samples, 3),
            dtype=torch.int64,
            generator=self.rng,
        )
        # Each corner in the cube is a different label
        labels = torch.einsum(
            "bi,i->b", coords, torch.tensor([1, 2, 4], dtype=torch.int64)
        )
        # Add Gaussian noise to coords
        coords = coords.float()
        coords += (
            torch.randn(self.n_samples, 3, dtype=torch.float32, generator=self.rng)
            * self.sigma
        )
        # The final features are the coordinates offset according to the labels, and uniform noise for all other features
        self.features = torch.zeros(
            self.n_samples, self.n_features, dtype=torch.float32
        )
        for i in range(self.n_samples):
            offset: int = int(labels[i].item())
            self.features[i, offset : offset + 3] += coords[i]
            # uniform noise on all other features
            self.features[i, :offset] = torch.rand(
                (1, offset), dtype=torch.float32, generator=self.rng
            )
            self.features[i, offset + 3 :] = torch.rand(
                (1, self.n_features - offset - 3),
                dtype=torch.float32,
                generator=self.rng,
            )
        # Convert labels to one-hot encoding
        self.labels = torch.nn.functional.one_hot(labels, num_classes=8).float()

    @override
    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]

    @override
    def __len__(self):
        return len(self.features)

    @override
    def get_all_data(self) -> tuple[MaskedFeatures, FeatureMask]:
        return self.features, self.labels

    @override
    def save(self, path: Path) -> None:
        torch.save(
            {
                "features": self.features,
                "labels": self.labels,
                "config": {
                    "n_samples": self.n_samples,
                    "seed": self.seed,
                    "sigma": self.sigma,
                },
            },
            path,
        )

    @classmethod
    @override
    def load(cls, path: Path) -> Self:
        data = torch.load(path)
        dataset = cls(**data["config"])
        dataset.features = data["features"]
        dataset.labels = data["labels"]
        return dataset


@final
class CubeDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    """
    The Cube dataset, as described in the paper "ODIN: Optimal Discovery of High-value INformation Using Model-based Deep Reinforcement Learning"

    Implements the AFADataset protocol.
    """

    n_classes = 8
    n_features = 20  # Fixed number of features

    def __init__(
        self,
        n_samples: int = 20000,
        seed: int = 123,
        non_informative_feature_mean: float = 0.5,
        informative_feature_std: float = 0.1,
        non_informative_feature_std: float = 0.3,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.seed = seed
        self.non_informative_feature_mean = non_informative_feature_mean
        self.non_informative_feature_std = non_informative_feature_std
        self.informative_feature_std = informative_feature_std

        # Constants
        self.n_cube_features = 10  # Number of cube features
        self.n_dummy_features = (
            self.n_features - self.n_cube_features
        )  # Remaining features are dummy features

        self.rng = torch.Generator()

    @override
    def generate_data(self) -> None:
        self.rng.manual_seed(self.seed)

        # Draw labels
        y_int = torch.randint(
            0, self.n_classes, (self.n_samples,), dtype=torch.int64, generator=self.rng
        )

        # Binary codes for labels (8×3)
        binary_codes = torch.stack(
            [
                torch.tensor([int(b) for b in format(i, "03b")])
                for i in range(self.n_classes)
            ],
            dim=0,
        ).flip(-1)

        # Initialize feature blocks
        X_cube = torch.normal(
            mean=self.non_informative_feature_mean,
            std=self.non_informative_feature_std,
            size=(self.n_samples, self.n_cube_features),
            generator=self.rng,
        )

        X_dummy = torch.normal(
            mean=self.non_informative_feature_mean,
            std=self.non_informative_feature_std,
            size=(self.n_samples, self.n_dummy_features),
            generator=self.rng,
        )

        # Insert informative signals
        for i in range(self.n_samples):
            lbl = int(y_int[i].item())
            mu_bin = binary_codes[lbl]

            # Cube features: 3 bumps
            idxs = [(lbl + j) for j in range(3)]
            X_cube[i, idxs] = (
                torch.normal(
                    mean=0.0,
                    std=self.informative_feature_std,
                    size=(3,),
                    generator=self.rng,
                )
                + mu_bin
            )

        # Concatenate all features
        self.features = torch.cat([X_cube, X_dummy], dim=1)
        assert self.features.shape[1] == self.n_features

        # Labels
        self.labels = y_int
        self.labels = torch.nn.functional.one_hot(
            self.labels, num_classes=self.n_classes
        ).float()
        assert self.labels.shape[1] == self.n_classes

    @override
    def __getitem__(self, idx: int) -> tuple[MaskedFeatures, FeatureMask]:
        return self.features[idx], self.labels[idx]

    @override
    def __len__(self):
        return len(self.features)

    @override
    def get_all_data(self) -> tuple[MaskedFeatures, FeatureMask]:
        return self.features, self.labels

    @override
    def save(self, path: Path) -> None:
        torch.save(
            {
                "features": self.features,
                "labels": self.labels,
                "config": {
                    "n_samples": self.n_samples,
                    "seed": self.seed,
                    "non_informative_feature_mean": self.non_informative_feature_mean,
                    "non_informative_feature_std": self.non_informative_feature_std,
                    "informative_feature_std": self.informative_feature_std,
                },
            },
            path,
        )

    @classmethod
    @override
    def load(cls, path: Path) -> Self:
        data = torch.load(path)
        dataset = cls(**data["config"])
        dataset.features = data["features"]
        dataset.labels = data["labels"]
        return dataset


@final
class CubeSimpleDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    """
    A simplified version of the cube dataset, made for debugging purposes. Three features contain label information and three others are noise.

    Implements the AFADataset protocol.
    """

    n_classes = 8
    n_features = 6

    def __init__(
        self,
        n_samples: int = 20000,
        seed: int = 123,
        non_informative_feature_mean: float = 0.5,
        informative_feature_std: float = 0.1,
        non_informative_feature_std: float = 0.3,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.seed = seed
        self.non_informative_feature_mean = non_informative_feature_mean
        self.non_informative_feature_std = non_informative_feature_std
        self.informative_feature_std = informative_feature_std

        self.rng = torch.Generator()

    @override
    def generate_data(self) -> None:
        self.rng.manual_seed(self.seed)

        # Draw labels
        y_int = torch.randint(
            0, self.n_classes, (self.n_samples,), dtype=torch.int64, generator=self.rng
        )

        # Binary codes for labels (8×3)
        binary_codes = torch.stack(
            [
                torch.tensor([int(b) for b in format(i, "03b")])
                for i in range(self.n_classes)
            ],
            dim=0,
        ).flip(-1)

        # Initialize feature blocks
        self.features = torch.normal(
            mean=self.non_informative_feature_mean,
            std=self.non_informative_feature_std,
            size=(self.n_samples, 6),
            generator=self.rng,
        )

        # Insert informative signals
        for i in range(self.n_samples):
            lbl = int(y_int[i].item())
            mu_bin = binary_codes[lbl]

            # Cube features: 3 bumps
            self.features[i, [0, 2, 4]] = (
                torch.normal(
                    mean=0.0,
                    std=self.informative_feature_std,
                    size=(3,),
                    generator=self.rng,
                )
                + mu_bin
            )

        assert self.features.shape[1] == self.n_features

        # Labels
        self.labels = y_int
        self.labels = torch.nn.functional.one_hot(
            self.labels, num_classes=self.n_classes
        ).float()
        assert self.labels.shape[1] == self.n_classes

    @override
    def __getitem__(self, idx: int) -> tuple[MaskedFeatures, FeatureMask]:
        return self.features[idx], self.labels[idx]

    @override
    def __len__(self):
        return len(self.features)

    @override
    def get_all_data(self) -> tuple[MaskedFeatures, FeatureMask]:
        return self.features, self.labels

    @override
    def save(self, path: Path) -> None:
        torch.save(
            {
                "features": self.features,
                "labels": self.labels,
                "config": {
                    "n_samples": self.n_samples,
                    "seed": self.seed,
                    "non_informative_feature_mean": self.non_informative_feature_mean,
                    "non_informative_feature_std": self.non_informative_feature_std,
                    "informative_feature_std": self.informative_feature_std,
                },
            },
            path,
        )

    @classmethod
    @override
    def load(cls, path: Path) -> Self:
        data = torch.load(path)
        dataset = cls(**data["config"])
        dataset.features = data["features"]
        dataset.labels = data["labels"]
        return dataset


@final
class CubeOnlyInformativeDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    """
    A version of the cube dataset that only has the first 10 informative features.

    Implements the AFADataset protocol.
    """

    n_classes = 8
    n_features = 10  # Fixed number of features

    def __init__(
        self,
        n_samples: int = 20000,
        seed: int = 123,
        informative_feature_std: float = 0.1,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.seed = seed
        self.informative_feature_std = informative_feature_std

        self.rng = torch.Generator()

    @override
    def generate_data(self) -> None:
        self.rng.manual_seed(self.seed)

        # Draw labels
        y_int = torch.randint(
            0, self.n_classes, (self.n_samples,), dtype=torch.int64, generator=self.rng
        )

        # Binary codes for labels (8×3)
        binary_codes = torch.stack(
            [
                torch.tensor([int(b) for b in format(i, "03b")])
                for i in range(self.n_classes)
            ],
            dim=0,
        ).flip(-1)

        # Initialize feature blocks
        self.features = torch.zeros(
            size=(self.n_samples, self.n_features),
        )

        # Insert informative signals
        for i in range(self.n_samples):
            lbl = int(y_int[i].item())
            mu_bin = binary_codes[lbl]

            # Cube features: 3 bumps
            idxs = [(lbl + j) for j in range(3)]
            self.features[i, idxs] = (
                torch.normal(
                    mean=0.0,
                    std=self.informative_feature_std,
                    size=(3,),
                    generator=self.rng,
                )
                + mu_bin
            )

        assert self.features.shape[1] == self.n_features

        # Labels
        self.labels = y_int
        self.labels = torch.nn.functional.one_hot(
            self.labels, num_classes=self.n_classes
        ).float()
        assert self.labels.shape[1] == self.n_classes

    @override
    def __getitem__(self, idx: int) -> tuple[MaskedFeatures, FeatureMask]:
        return self.features[idx], self.labels[idx]

    @override
    def __len__(self):
        return len(self.features)

    @override
    def get_all_data(self) -> tuple[MaskedFeatures, FeatureMask]:
        return self.features, self.labels

    @override
    def save(self, path: Path) -> None:
        torch.save(
            {
                "features": self.features,
                "labels": self.labels,
                "config": {
                    "n_samples": self.n_samples,
                    "seed": self.seed,
                    "informative_feature_std": self.informative_feature_std,
                },
            },
            path,
        )

    @classmethod
    @override
    def load(cls, path: Path) -> Self:
        data = torch.load(path)
        dataset = cls(**data["config"])
        dataset.features = data["features"]
        dataset.labels = data["labels"]
        return dataset


@final
class AFAContextDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    """
    A hybrid dataset combining context-based feature selection and the Cube dataset.

    - Features:
        * First 3 features: one-hot context (0, 1, or 2)
        * Next 10 features: Cube features if context == 0, else noise
        * Next 10 features: Cube features if context == 1, else noise
        * Next 10 features: Cube features if context == 2, else noise

    - Label:
        * One of 8 classes encoded by a 3-bit binary vector inserted into the relevant block

    Optimal policy: query the context first, then only the relevant 10-dimensional block.
    """

    n_classes = 8
    n_contexts = 5
    n_features = 3 + 5 * 10  # 5 context + 50 cube features
    block_size = 10

    def __init__(
        self,
        n_samples: int = 20000,
        seed: int = 123,
        context_feature_std: float = 0.1,
        informative_feature_std: float = 0.1,
        non_informative_feature_mean: float = 0.5,
        non_informative_feature_std: float = 0.3,
    ):
        self.n_samples = n_samples
        self.seed = seed
        self.context_feature_std = context_feature_std
        self.informative_feature_std = informative_feature_std
        self.non_informative_feature_mean = non_informative_feature_mean
        self.non_informative_feature_std = non_informative_feature_std

        self.rng = torch.Generator().manual_seed(seed)
        self.features: Tensor
        self.labels: Tensor

        self.generate_data()

    @override
    def generate_data(self) -> None:
        # Sample context (0, 1, 2)
        context = torch.randint(
            0, self.n_contexts, (self.n_samples,), generator=self.rng
        )
        context_onehot = F.one_hot(
            context, num_classes=self.n_contexts
        ).float() + torch.normal(
            mean=0,
            std=self.context_feature_std,
            size=(self.n_samples, self.n_contexts),
            generator=self.rng,
        )  # (n_samples, 5)

        # Sample labels 0–7
        y_int = torch.randint(0, self.n_classes, (self.n_samples,), generator=self.rng)

        # Binary codes for labels (8×3)
        binary_codes = torch.stack(
            [
                torch.tensor([int(b) for b in format(i, "03b")])
                for i in range(self.n_classes)
            ],
            dim=0,
        ).flip(-1)  # (8, 3)

        # Create 3 blocks of features, each 10D
        blocks = []
        for block_context in range(self.n_contexts):
            block = torch.normal(
                mean=self.non_informative_feature_mean,
                std=self.non_informative_feature_std,
                size=(self.n_samples, self.block_size),
                generator=self.rng,
            )
            blocks.append(block)
        blocks = torch.stack(blocks, dim=1)  # (n_samples, 3, 10)

        # Insert informative signal into the correct block based on context
        for i in range(self.n_samples):
            ctx = int(context[i].item())
            label = int(y_int[i].item())
            bin_code = binary_codes[label].float()

            # Select 3 indices in the block to hold the binary code (arbitrary positions)
            insert_idx = [(label + j) % self.block_size for j in range(3)]

            noise = torch.normal(
                mean=0.0,
                std=self.informative_feature_std,
                size=(3,),
                generator=self.rng,
            )
            blocks[i, ctx, insert_idx] = bin_code + noise

        # Flatten blocks: (n_samples, 50)
        block_features = blocks.view(self.n_samples, -1)

        # Final feature matrix: context (5) + all block features (50)
        self.features = torch.cat(
            [context_onehot, block_features], dim=1
        )  # (n_samples, 33)

        # One-hot labels
        self.labels = F.one_hot(y_int, num_classes=self.n_classes).float()

    @override
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.features[idx], self.labels[idx]

    @override
    def __len__(self) -> int:
        return self.features.size(0)

    @override
    def get_all_data(self) -> tuple[Tensor, Tensor]:
        return self.features, self.labels

    @override
    def save(self, path: Path) -> None:
        torch.save(
            {
                "features": self.features,
                "labels": self.labels,
                "config": {
                    "n_samples": self.n_samples,
                    "seed": self.seed,
                    "informative_feature_std": self.informative_feature_std,
                    "non_informative_feature_mean": self.non_informative_feature_mean,
                    "non_informative_feature_std": self.non_informative_feature_std,
                },
            },
            path,
        )

    @classmethod
    @override
    def load(cls, path: Path):
        data = torch.load(path)
        dataset = cls(**data["config"])
        dataset.features = data["features"]
        dataset.labels = data["labels"]
        return dataset


@final
class ContextSelectiveXORDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    """
    Each data point:
      - context c ∈ {0, 1}
      - features: x1, x2, x3, x4 ∈ {0, 1}

    Label:
      - if c == 0: label = x1 XOR x2
      - if c == 1: label = x3 XOR x4

    Optimal strategy: query context first, then only relevant pair.
    """

    n_classes = 2
    n_features = 5

    def __init__(self, n_samples: int = 1000, seed: int = 42):
        super().__init__()
        self.n_samples = n_samples
        self.seed = seed
        self.rng = torch.Generator().manual_seed(seed)
        self.features: Tensor
        self.labels: Tensor

        self.generate_data()

    @override
    def generate_data(self) -> None:
        # Sample context bit: 0 or 1
        context = torch.randint(0, 2, (self.n_samples,), generator=self.rng)

        # Sample binary features: x1, x2, x3, x4
        x = torch.randint(0, 2, (self.n_samples, 4), generator=self.rng)

        labels = torch.zeros(self.n_samples, dtype=torch.int64)
        for i in range(self.n_samples):
            if context[i] == 0:
                labels[i] = int(x[i, 0] ^ x[i, 1])  # x1 XOR x2
            else:
                labels[i] = int(x[i, 2] ^ x[i, 3])  # x3 XOR x4

        # One-hot labels
        self.labels = torch.nn.functional.one_hot(labels, num_classes=2).float()

        # Combine context and features into full feature vector
        self.features = torch.cat(
            [
                context.unsqueeze(1).float(),  # shape: (n_samples, 1)
                x.float(),  # shape: (n_samples, 4)
            ],
            dim=1,
        )  # shape: (n_samples, 5)

    @override
    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]

    @override
    def __len__(self):
        return self.features.size(0)

    @override
    def get_all_data(self):
        return self.features, self.labels

    @override
    def save(self, path: Path) -> None:
        torch.save(
            {
                "features": self.features,
                "labels": self.labels,
                "config": {
                    "n_samples": self.n_samples,
                    "seed": self.seed,
                },
            },
            path,
        )

    @classmethod
    @override
    def load(cls, path: Path) -> Self:
        data = torch.load(path)
        cfg = data["config"]
        ds = cls(**cfg)
        ds.features = data["features"]
        ds.labels = data["labels"]
        return ds


@final
class MNISTDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    """MNIST dataset wrapped to follow the AFADataset protocol."""

    n_classes = 10
    n_features = 784  # Fixed number of features (28x28 images flattened)

    def __init__(
        self,
        train: bool = True,
        transform: Callable[[Tensor], Tensor] | None = None,
        download: bool = True,
        root: str = "data/MNIST",
        seed: None = None,  # does nothing, added to not panic during data generation
    ):
        super().__init__()
        self.train = train
        self.transform = transform if transform is not None else transforms.ToTensor()
        self.download = download
        self.root = root

        self.dataset = None  # set when generating data

    @override
    def generate_data(self) -> None:
        self.dataset = datasets.MNIST(
            root=self.root,
            train=self.train,
            transform=self.transform,
            download=self.download,
        )
        # Convert images to features (flatten)
        self.features = torch.stack([x[0].flatten() for x in self.dataset]).float()
        assert self.features.shape[1] == self.n_features
        self.labels = torch.tensor([x[1] for x in self.dataset])
        self.labels = torch.nn.functional.one_hot(
            self.labels, num_classes=self.n_classes
        ).float()
        assert self.labels.shape[1] == self.n_classes

    @override
    def __getitem__(self, idx: int) -> tuple[Features, Label]:
        return self.features[idx], self.labels[idx]

    @override
    def __len__(self):
        return self.features.size(0)

    @override
    def get_all_data(self) -> tuple[Features, Label]:
        return self.features, self.labels

    @override
    def save(self, path: Path) -> None:
        torch.save(
            {
                "features": self.features,
                "labels": self.labels,
                "config": {
                    "train": self.train,
                    "root": self.root,
                },
            },
            path,
        )

    @classmethod
    @override
    def load(cls, path: Path) -> Self:
        data = torch.load(path)
        dataset = cls(**data["config"])
        dataset.features = data["features"]
        dataset.labels = data["labels"]
        return dataset


@final
class FashionMNISTDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    """FashionMNIST dataset wrapped to follow the AFADataset protocol."""

    n_classes = 10
    n_features = 784  # 28x28 flattened images

    def __init__(
        self,
        train: bool = True,
        transform: Callable[[Tensor], Tensor] | None = None,
        download: bool = True,
        root: str = "data/FashionMNIST",
        seed: None = None,  # does nothing, added to not panic during data generation
    ):
        super().__init__()
        self.train = train
        self.transform = transform if transform is not None else transforms.ToTensor()
        self.download = download
        self.root = root
        self.dataset = None  # set when generating data

    @override
    def generate_data(self) -> None:
        self.dataset = datasets.FashionMNIST(
            root=self.root,
            train=self.train,
            transform=self.transform,
            download=self.download,
        )
        # Convert images to flattened feature vectors
        self.features = torch.stack([x[0].flatten() for x in self.dataset]).float()
        assert self.features.shape[1] == self.n_features
        self.labels = torch.tensor([x[1] for x in self.dataset])
        self.labels = torch.nn.functional.one_hot(
            self.labels, num_classes=self.n_classes
        ).float()
        assert self.labels.shape[1] == self.n_classes

    @override
    def __getitem__(self, idx: int) -> tuple[Features, Label]:
        return self.features[idx], self.labels[idx]

    @override
    def __len__(self):
        return self.features.size(0)

    @override
    def get_all_data(self) -> tuple[Features, Label]:
        return self.features, self.labels

    @override
    def save(self, path: Path) -> None:
        torch.save(
            {
                "features": self.features,
                "labels": self.labels,
                "config": {
                    "train": self.train,
                    "root": self.root,
                },
            },
            path,
        )

    @classmethod
    @override
    def load(cls, path: Path) -> Self:
        data = torch.load(path)
        dataset = cls(**data["config"])
        dataset.features = data["features"]
        dataset.labels = data["labels"]
        return dataset


@final
class DiabetesDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    """
    Diabetes dataset wrapped to follow the AFADataset protocol.

    This dataset contains medical measurements and indicators for diabetes classification.
    The target variable has 3 classes (0, 1, 2) representing different diabetes outcomes.
    """

    n_classes = 3
    n_features = 45

    def __init__(
        self,
        data_path: str = "datasets/diabetes.csv",
        seed: int = 123,
    ):
        super().__init__()
        self.data_path = data_path
        self.seed = seed
        self.feature_names = None  # set when generating data

    @override
    def generate_data(self) -> None:
        """Load and preprocess the diabetes dataset."""
        # Set random seed for reproducibility
        torch.manual_seed(self.seed)

        # Check if file exists
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Diabetes dataset not found at {self.data_path}")

        # Load the dataset
        df = pd.read_csv(self.data_path)

        # Extract features and labels
        # The last column is the target variable (Outcome)
        features_df = df.iloc[:, :-1]
        labels_df = df.iloc[:, -1]

        # Convert to tensors
        self.features = torch.tensor(features_df.values, dtype=torch.float32)
        assert self.features.shape[1] == self.n_features
        self.labels = torch.tensor(labels_df.values, dtype=torch.long)
        self.labels = torch.nn.functional.one_hot(
            self.labels, num_classes=self.n_classes
        ).float()
        assert self.labels.shape[1] == self.n_classes

        # Store feature names
        self.feature_names = features_df.columns.tolist()

    @override
    def __getitem__(self, idx: int) -> tuple[Features, Label]:
        """Return a single sample from the dataset."""
        return self.features[idx], self.labels[idx]

    @override
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.features)

    @override
    def get_all_data(self) -> tuple[Features, Label]:
        """Return all features and labels."""
        return self.features, self.labels

    @override
    def save(self, path: Path) -> None:
        """Save the dataset to a file."""
        torch.save(
            {
                "features": self.features,
                "labels": self.labels,
                "feature_names": self.feature_names,
                "config": {
                    "data_path": self.data_path,
                    "seed": self.seed,
                },
            },
            path,
        )

    @classmethod
    @override
    def load(cls, path: Path) -> Self:
        """Load a dataset from a file."""
        data = torch.load(path)
        dataset = cls(**data["config"])
        dataset.features = data["features"]
        dataset.labels = data["labels"]
        dataset.feature_names = data["feature_names"]
        return dataset


@final
class MiniBooNEDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    """
    MiniBooNE dataset wrapped to follow the AFADataset protocol.

    This dataset contains particle physics measurements from the MiniBooNE experiment.
    The target variable has 2 classes (signal and background).
    """

    n_classes = 2
    n_features = 50

    def __init__(
        self,
        data_path: str = "datasets/miniboone.csv",
        seed: int = 123,
    ):
        super().__init__()
        self.data_path = data_path
        self.seed = seed
        self.feature_names = None  # set when generating data

    @override
    def generate_data(self) -> None:
        """Load and preprocess the MiniBooNE dataset."""
        torch.manual_seed(self.seed)

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"MiniBooNE dataset not found at {self.data_path}")

        df = pd.read_csv(self.data_path)

        # Assuming the last column is the binary target variable
        features_df = df.iloc[:, :-1]
        labels_df = df.iloc[:, -1]

        self.features = torch.tensor(features_df.values, dtype=torch.float32)
        assert self.features.shape[1] == self.n_features

        self.labels = torch.tensor(labels_df.values, dtype=torch.long)
        self.labels = torch.nn.functional.one_hot(
            self.labels, num_classes=self.n_classes
        ).float()
        assert self.labels.shape[1] == self.n_classes

        self.feature_names = features_df.columns.tolist()

    @override
    def __getitem__(self, idx: int) -> tuple[Features, Label]:
        return self.features[idx], self.labels[idx]

    @override
    def __len__(self) -> int:
        return len(self.features)

    @override
    def get_all_data(self) -> tuple[Features, Label]:
        return self.features, self.labels

    @override
    def save(self, path: Path) -> None:
        torch.save(
            {
                "features": self.features,
                "labels": self.labels,
                "feature_names": self.feature_names,
                "config": {
                    "data_path": self.data_path,
                    "seed": self.seed,
                },
            },
            path,
        )

    @classmethod
    @override
    def load(cls, path: Path) -> Self:
        data = torch.load(path)
        dataset = cls(**data["config"])
        dataset.features = data["features"]
        dataset.labels = data["labels"]
        dataset.feature_names = data["feature_names"]
        return dataset


@final
class PhysionetDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    """
    Physionet dataset wrapped to follow the AFADataset protocol.

    This dataset contains medical measurements from ICU patients.
    The target variable has 2 classes (0, 1) representing different outcomes.
    """

    n_classes = 2
    n_features = 41

    def __init__(
        self,
        data_path: str = "datasets/physionet_data.csv",
        seed: int = 123,
    ):
        super().__init__()
        self.data_path = data_path
        self.seed = seed
        self.feature_names = None  # set when generating data

    @override
    def generate_data(self) -> None:
        """Load and preprocess the Physionet dataset."""
        torch.manual_seed(self.seed)

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Physionet dataset not found at {self.data_path}")

        df = pd.read_csv(self.data_path)

        features_df = df.iloc[:, :-1]
        labels_df = df.iloc[:, -1]

        # Handle missing values by filling with column means
        features_df = features_df.fillna(features_df.mean())

        # Convert to tensors
        # self.features = torch.tensor(scaled_features, dtype=torch.float32)
        self.features = torch.tensor(features_df.values, dtype=torch.float32)

        # Check for NaNs after tensor conversion
        if torch.isnan(self.features).any():
            raise ValueError("NaNs detected in features after filling missing values.")

        # === Standardize features ===
        # scaler = StandardScaler()
        # scaled_features = scaler.fit_transform(features_df.values)

        assert self.features.shape[1] == self.n_features

        self.labels = torch.tensor(labels_df.values, dtype=torch.long)
        self.labels = torch.nn.functional.one_hot(
            self.labels, num_classes=self.n_classes
        ).float()
        assert self.labels.shape[1] == self.n_classes

        self.feature_names = features_df.columns.tolist()

    @override
    def __getitem__(self, idx: int) -> tuple[Features, Label]:
        return self.features[idx], self.labels[idx]

    @override
    def __len__(self) -> int:
        return len(self.features)

    @override
    def get_all_data(self) -> tuple[Features, Label]:
        return self.features, self.labels

    @override
    def save(self, path: Path) -> None:
        torch.save(
            {
                "features": self.features,
                "labels": self.labels,
                "feature_names": self.feature_names,
                "config": {
                    "data_path": self.data_path,
                    "seed": self.seed,
                },
            },
            path,
        )

    @classmethod
    @override
    def load(cls, path: Path) -> Self:
        data = torch.load(path)
        dataset = cls(**data["config"])
        dataset.features = data["features"]
        dataset.labels = data["labels"]
        dataset.feature_names = data["feature_names"]
        return dataset
