import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import math
import pandas as pd
import os

from common.custom_types import FeatureMask, MaskedFeatures, Features, Label

class CubeDataset(Dataset):
    """
    The Cube dataset, as described in the paper "ODIN: Optimal Discovery of High-value INformation Using Model-based Deep Reinforcement Learning"

    Implements the AFADataset protocol.
    """

    n_classes = 8

    def __init__(
        self,
        n_features: int = 20,
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
        self.n_classes = 8
        self.n_cube_features = 10  # Number of cube features
        self.n_dummy_features = n_features - self.n_cube_features  # Remaining features are dummy features
        
        # Placeholder attributes
        self.features = None
        self.labels = None
        self.feature_names = None
        
    def generate_data(self) -> None:
        rng = torch.Generator()
        rng.manual_seed(self.seed)
        
        # Draw labels
        y_int = torch.randint(0, self.n_classes, (self.n_samples,), dtype=torch.int64, generator=rng)
        
        # Binary codes for labels (8×3)
        binary_codes = torch.stack([
            torch.tensor([int(b) for b in format(i, '03b')])
            for i in range(self.n_classes)
        ], dim=0).flip(-1)
        
        # Initialize feature blocks
        X_cube = torch.normal(
            mean=self.non_informative_feature_mean,
            std=self.non_informative_feature_std,
            size=(self.n_samples, self.n_cube_features),
            generator=rng,
        )

        X_dummy = torch.normal(
            mean=self.non_informative_feature_mean,
            std=self.non_informative_feature_std,
            size=(self.n_samples, self.n_dummy_features),
            generator=rng,
        )
        
        # Insert informative signals
        for i in range(self.n_samples):
            lbl = y_int[i].item()
            mu_bin = binary_codes[lbl]
            
            # Cube features: 3 bumps
            idxs = [(lbl + j) for j in range(3)]
            X_cube[i, idxs] = torch.normal(
                mean=0.0,
                std=self.informative_feature_std,
                size=(3,),
                generator=rng
            ) + mu_bin

        # Concatenate all features
        self.features = torch.cat([X_cube, X_dummy], dim=1)
        
        # Labels
        self.labels = y_int
        self.labels = torch.nn.functional.one_hot(self.labels, num_classes=self.n_classes).float()

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
                    "n_features": self.n_cube_features + self.n_dummy_features,
                    "n_samples": self.n_samples,
                    "seed": self.seed,
                    "non_informative_feature_mean": self.non_informative_feature_mean,
                    "non_informative_feature_std": self.non_informative_feature_std,
                    "informative_feature_std": self.informative_feature_std,
                },
            },
            path,
        )

    @staticmethod
    def load(path: str) -> "CubeDataset":
        data = torch.load(path)
        dataset = CubeDataset(**data["config"])
        dataset.features = data["features"]
        dataset.labels = data["labels"]
        return dataset

class AFAContextDataset(Dataset):
    """
    A PyTorch Dataset merging AFA structure with cube-dataset dummy-feature behavior.

    Implements the AFADataset protocol.
    """
    
    n_classes = 8

    def __init__(
        self,
        n_samples: int = 1000,
        std_bin: float = 0.1,
        std_cube: float = 1.0,
        bin_feature_cost: float = 5.0,
        n_dummy_features: int = 10,
        seed: int = 123,
        non_informative_feature_mean: float = 0.5,
        non_informative_feature_std: float = 0.3,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.std_bin = std_bin
        self.std_cube = std_cube
        self.bin_feature_cost = bin_feature_cost
        self.n_dummy_features = n_dummy_features
        self.seed = seed
        self.non_informative_feature_mean = non_informative_feature_mean
        self.non_informative_feature_std = non_informative_feature_std

        # Constants
        self.n_classes = 8
        self.n_context_groups = 3
        self.group_size = 3
        self.n_bin_features = self.n_context_groups * self.group_size
        self.n_cube_features = 10

        # Placeholder attributes
        self.features = None
        self.labels = None
        self.costs = None

    def generate_data(self) -> None:
        rng = torch.Generator()
        rng.manual_seed(self.seed)

        # Draw labels and context
        y_int = torch.randint(0, self.n_classes, (self.n_samples,), dtype=torch.int64, generator=rng)
        S = torch.randint(0, self.n_context_groups, (self.n_samples,), dtype=torch.int64, generator=rng)

        # Binary codes for labels (8×3)
        binary_codes = torch.stack([
            torch.tensor([int(b) for b in format(i, '03b')])
            for i in range(self.n_classes)
        ], dim=0).flip(-1)

        # Initialize feature blocks
        X_context = S.unsqueeze(1).float()

        X_bin = torch.normal(
            mean=self.non_informative_feature_mean,
            std=self.non_informative_feature_std,
            size=(self.n_samples, self.n_bin_features),
            generator=rng,
        )

        X_cube = torch.normal(
            mean=self.non_informative_feature_mean,
            std=self.non_informative_feature_std,
            size=(self.n_samples, self.n_cube_features),
            generator=rng,
        )

        X_dummy = torch.normal(
            mean=self.non_informative_feature_mean,
            std=self.non_informative_feature_std,
            size=(self.n_samples, self.n_dummy_features),
            generator=rng,
        )

        # Insert informative signals
        for i in range(self.n_samples):
            lbl = y_int[i].item()
            ctx = S[i].item()
            mu_bin = binary_codes[lbl]

            # Binary features in active group
            start = ctx * self.group_size
            end = start + self.group_size
            X_bin[i, start:end] = torch.normal(
                mean=0.0,
                std=self.std_bin,
                size=(self.group_size,),
                generator=rng
            ) + mu_bin

            # Cube features: 3 bumps
            idxs = [(lbl + j) for j in range(3)]
            X_cube[i, idxs] = torch.normal(
                mean=0.0,
                std=self.std_cube,
                size=(3,),
                generator=rng
            ) + mu_bin

        # Concatenate all features
        self.features = torch.cat([X_context, X_bin, X_cube, X_dummy], dim=1)

        # Build costs vector
        total_dim = self.features.shape[1]
        costs = torch.ones(total_dim)
        costs[1:1 + self.n_bin_features] = self.bin_feature_cost
        self.costs = costs

        # One-hot labels
        self.labels = y_int
        self.labels = torch.nn.functional.one_hot(self.labels, num_classes=self.n_classes).float()

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]

    def __len__(self):
        return self.features.size(0)

    def get_all_data(self):
        return self.features, self.labels

    def save(self, path: str) -> None:
        torch.save(
            {
                'features': self.features,
                'labels': self.labels,
                'costs': self.costs,
                'config': {
                    'n_samples': self.n_samples,
                    'std_bin': self.std_bin,
                    'std_cube': self.std_cube,
                    'bin_feature_cost': self.bin_feature_cost,
                    'n_dummy_features': self.n_dummy_features,
                    'seed': self.seed,
                    'non_informative_feature_mean': self.non_informative_feature_mean,
                    'non_informative_feature_std': self.non_informative_feature_std,
                },
            },
            path,
        )

    @staticmethod
    def load(path: str) -> 'AFAContextDataset':
        data = torch.load(path)
        cfg = data['config']
        ds = AFAContextDataset(**cfg)
        ds.features = data['features']
        ds.labels = data['labels']
        ds.costs = data['costs']
        return ds

class MNISTDataset(Dataset):
    """
    MNIST dataset wrapped to follow the AFADataset protocol.
    """
    
    n_classes = 10

    def __init__(
        self,
        train: bool = True,
        transform=None,
        download: bool = True,
        root: str = "data/MNIST",
    ):
        super().__init__()
        self.train = train
        self.transform = transform if transform is not None else transforms.ToTensor()
        self.download = download
        self.n_classes = 10
        self.root = root
        self.dataset = None
        self.features = None
        self.labels = None

    def generate_data(self) -> None:
        self.dataset = datasets.MNIST(
            root=self.root,
            train=self.train,
            transform=self.transform,
            download=self.download
        )
        # Convert images to features (flatten)
        self.features = torch.stack([x[0].flatten() for x in self.dataset])
        # Convert labels to one-hot
        #self.labels = torch.nn.functional.one_hot(
        #    torch.tensor([x[1] for x in self.dataset]),
        #    num_classes=10
        #).float()
        self.labels = torch.tensor([x[1] for x in self.dataset])
        self.labels = torch.nn.functional.one_hot(self.labels, num_classes=self.n_classes).float()

    def __getitem__(self, idx: int) -> tuple[Features, Label]:
        return self.features[idx], self.labels[idx]

    def __len__(self):
        return self.features.size(0)

    def get_all_data(self) -> tuple[Features, Label]:
        return self.features, self.labels

    def save(self, path: str) -> None:
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

    @staticmethod
    def load(path: str) -> "MNISTDataset":
        data = torch.load(path)
        dataset = MNISTDataset(**data["config"])
        dataset.features = data["features"]
        dataset.labels = data["labels"]
        return dataset

class DiabetesDataset(Dataset):
    """
    Diabetes dataset wrapped to follow the AFADataset protocol.
    
    This dataset contains medical measurements and indicators for diabetes classification.
    The target variable has 3 classes (0, 1, 2) representing different diabetes outcomes.
    """
    
    n_classes = 3

    def __init__(
        self,
        data_path: str = "datasets/diabetes.csv",
        seed: int = 123,
    ):
        super().__init__()
        self.data_path = data_path
        self.seed = seed
        
        # Placeholder attributes
        self.features = None
        self.labels = None
        self.feature_names = None
        self.n_classes = 3
            
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
        self.features = torch.tensor(features_df.values)
        self.labels = torch.tensor(labels_df.values, dtype=torch.long)
        self.labels = torch.nn.functional.one_hot(self.labels, num_classes=self.n_classes).float()
        
        # Store feature names
        self.feature_names = features_df.columns.tolist()
        
        print(f"Loaded Diabetes dataset with {len(self.features)} samples and {self.features.shape[1]} features")
        #print(f"Class distribution: {torch.bincount(self.labels.long(), minlength=3)}")
    
    def __getitem__(self, idx: int) -> tuple[Features, Label]:
        """Return a single sample from the dataset."""
        return self.features[idx], self.labels[idx]
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.features)
    
    def get_all_data(self) -> tuple[Features, Label]:
        """Return all features and labels."""
        return self.features, self.labels
    
    def save(self, path: str) -> None:
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
    
    @staticmethod
    def load(path: str) -> "DiabetesDataset":
        """Load a dataset from a file."""
        data = torch.load(path)
        dataset = DiabetesDataset(**data["config"])
        dataset.features = data["features"]
        dataset.labels = data["labels"]
        dataset.feature_names = data["feature_names"]
        return dataset

class PhysionetDataset(Dataset):
    """
    Physionet dataset wrapped to follow the AFADataset protocol.
    
    This dataset contains medical measurements from ICU patients.
    The target variable has 2 classes (0, 1) representing different outcomes.
    """

    n_classes = 2

    def __init__(
        self,
        data_path: str = "datasets/physionet_data.csv",
        seed: int = 123,
    ):
        super().__init__()
        self.data_path = data_path
        self.seed = seed
        
        # Placeholder attributes
        self.features = None
        self.labels = None
        self.feature_names = None
        self.n_classes = 2
            
    def generate_data(self) -> None:
        """Load and preprocess the Physionet dataset."""
        # Set random seed for reproducibility
        torch.manual_seed(self.seed)
        
        # Check if file exists
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Physionet dataset not found at {self.data_path}")
        
        # Load the dataset
        df = pd.read_csv(self.data_path)
        
        # Extract features and labels
        # The last column is the target variable (Outcome)
        features_df = df.iloc[:, :-1]
        labels_df = df.iloc[:, -1]
        
        # Handle missing values by filling with column means
        features_df = features_df.fillna(features_df.mean())
        
        # Convert to tensors
        self.features = torch.tensor(features_df.values)
        # Convert labels to LongTensor for one_hot encoding
        self.labels = torch.tensor(labels_df.values, dtype=torch.long)
        self.labels = torch.nn.functional.one_hot(self.labels, num_classes=self.n_classes).float()

        # Store feature names
        self.feature_names = features_df.columns.tolist()
        
        print(f"Loaded Physionet dataset with {len(self.features)} samples and {self.features.shape[1]} features")
        #print(f"Class distribution: {torch.bincount(self.labels, minlength=2)}")
    
    def __getitem__(self, idx: int) -> tuple[Features, Label]:
        """Return a single sample from the dataset."""
        return self.features[idx], self.labels[idx]
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.features)
    
    def get_all_data(self) -> tuple[Features, Label]:
        """Return all features and labels."""
        return self.features, self.labels
    
    def save(self, path: str) -> None:
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
    
    @staticmethod
    def load(path: str) -> "PhysionetDataset":
        """Load a dataset from a file."""
        data = torch.load(path)
        dataset = PhysionetDataset(**data["config"])
        dataset.features = data["features"]
        dataset.labels = data["labels"]
        dataset.feature_names = data["feature_names"]
        return dataset

#class CubeDatasetOld(Dataset):
#    """
#    The Cube dataset, as described in the paper "Minimizing data consumption with sequential online feature selection" (https://doi.org/10.1007/s13042-012-0092-x).
#
#    Implements the AFADataset protocol.
#    """
#
#    def __init__(
#        self,
#        n_features: int = 20,
#        data_points: int = 20000,
#        sigma: float = 0.1,
#        seed: int = 123,
#    ):
#        super().__init__()
#        self.n_features = n_features
#        self.data_points = data_points
#        self.sigma = sigma
#        self.seed = seed
#
#    def generate_data(self) -> None:
#        rng = torch.Generator()
#        rng.manual_seed(self.seed)
#        # Each coordinate is drawn from a Bernoulli distribution with p=0.5, which is the same as uniform
#        coords: Float[Tensor, "{data_points} 3"] = torch.randint(
#            low=0, high=2, size=(self.data_points, 3), dtype=torch.int64, generator=rng
#        )
#        # Each corner in the cube is a different label
#        labels = torch.einsum(
#            "bi,i->b", coords, torch.tensor([1, 2, 4], dtype=torch.int64)
#        )
#        # Add Gaussian noise to coords
#        coords = coords.float()
#        coords += (
#            torch.randn(self.data_points, 3, dtype=torch.float32, generator=rng)
#            * self.sigma
#        )
#        # The final features are the coordinates offset according to the labels, and uniform noise for all other features
#        self.features = torch.zeros(
#            self.data_points, self.n_features, dtype=torch.float32
#        )
#        for i in range(self.data_points):
#            offset: int = labels[i].item()
#            self.features[i, offset : offset + 3] += coords[i]
#            # uniform noise on all other features
#            self.features[i, :offset] = torch.rand(
#                (1, offset), dtype=torch.float32, generator=rng
#            )
#            self.features[i, offset + 3 :] = torch.rand(
#                (1, self.n_features - offset - 3), dtype=torch.float32, generator=rng
#            )
#        # Convert labels to one-hot encoding
#        self.labels = torch.nn.functional.one_hot(labels, num_classes=8).float()
#
#    def __getitem__(self, idx: int) -> tuple[MaskedFeatures, FeatureMask]:
#        return self.features[idx], self.labels[idx]
#
#    def __len__(self):
#        return len(self.features)
#
#    def get_all_data(self) -> tuple[MaskedFeatures, FeatureMask]:
#        return self.features, self.labels
#
#    def save(self, path: str) -> None:
#        torch.save(
#            {
#                "features": self.features,
#                "labels": self.labels,
#                "config": {
#                    "n_features": self.n_features,
#                    "data_points": self.data_points,
#                    "sigma": self.sigma,
#                    "seed": self.seed,
#                },
#            },
#            path,
#        )
#
#    @staticmethod
#    def load(path: str) -> "CubeDataset":
#        data = torch.load(path)
#        dataset = CubeDataset(**data["config"])
#        dataset.features = data["features"]
#        dataset.labels = data["labels"]
#        return dataset
