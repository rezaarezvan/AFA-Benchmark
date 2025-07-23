from pathlib import Path
from typing import Callable, Self, final, override
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import pandas as pd
import os


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
class AFAContextDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    """
    A PyTorch Dataset merging AFA structure with cube-dataset dummy-feature behavior.

    Implements the AFADataset protocol.
    """

    n_classes = 8
    n_features = 30  # Fixed number of features

    def __init__(
        self,
        n_samples: int = 1000,
        std_bin: float = 0.1,
        std_cube: float = 1.0,
        bin_feature_cost: float = 5.0,
        seed: int = 123,
        non_informative_feature_mean: float = 0.5,
        non_informative_feature_std: float = 0.3,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.std_bin = std_bin
        self.std_cube = std_cube
        self.bin_feature_cost = bin_feature_cost
        self.seed = seed
        self.non_informative_feature_mean = non_informative_feature_mean
        self.non_informative_feature_std = non_informative_feature_std
        self.costs = None  # set when generating data

        # Constants
        self.n_context_groups = 3
        self.group_size = 3
        self.n_bin_features = self.n_context_groups * self.group_size  # 9
        self.n_cube_features = 10
        self.n_dummy_features = self.n_features - (
            1 + self.n_bin_features + self.n_cube_features
        )

        self.rng = torch.Generator()

    @override
    def generate_data(self) -> None:
        self.rng.manual_seed(self.seed)

        # Draw labels and context
        y_int = torch.randint(
            0, self.n_classes, (self.n_samples,), dtype=torch.int64, generator=self.rng
        )
        S = torch.randint(
            0,
            self.n_context_groups,
            (self.n_samples,),
            dtype=torch.int64,
            generator=self.rng,
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
        X_context = S.unsqueeze(1).float()

        X_bin = torch.normal(
            mean=self.non_informative_feature_mean,
            std=self.non_informative_feature_std,
            size=(self.n_samples, self.n_bin_features),
            generator=self.rng,
        )

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
            ctx = S[i].item()
            mu_bin = binary_codes[lbl]

            # Binary features in active group
            start = ctx * self.group_size
            end = start + self.group_size
            X_bin[i, start:end] = (
                torch.normal(
                    mean=0.0,
                    std=self.std_bin,
                    size=(self.group_size,),
                    generator=self.rng,
                )
                + mu_bin
            )

            # Cube features: 3 bumps
            idxs = [(lbl + j) for j in range(3)]
            X_cube[i, idxs] = (
                torch.normal(mean=0.0, std=self.std_cube, size=(3,), generator=self.rng)
                + mu_bin
            )

        # Concatenate all features
        self.features = torch.cat([X_context, X_bin, X_cube, X_dummy], dim=1).float()
        assert self.features.shape[1] == self.n_features

        # Build costs vector
        total_dim = self.features.shape[1]
        costs = torch.ones(total_dim)
        costs[1 : 1 + self.n_bin_features] = self.bin_feature_cost
        self.costs = costs

        # One-hot labels
        self.labels = y_int
        self.labels = torch.nn.functional.one_hot(
            self.labels, num_classes=self.n_classes
        ).float()
        assert self.labels.shape[1] == self.n_classes

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
                "costs": self.costs,
                "config": {
                    "n_samples": self.n_samples,
                    "std_bin": self.std_bin,
                    "std_cube": self.std_cube,
                    "bin_feature_cost": self.bin_feature_cost,
                    "seed": self.seed,
                    "non_informative_feature_mean": self.non_informative_feature_mean,
                    "non_informative_feature_std": self.non_informative_feature_std,
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
        ds.costs = data["costs"]
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

        # === Standardize features ===
        # scaler = StandardScaler()
        # scaled_features = scaler.fit_transform(features_df.values)

        # Convert to tensors
        # self.features = torch.tensor(scaled_features, dtype=torch.float32)
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
