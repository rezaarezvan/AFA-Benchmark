import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import Dataset

from common.custom_types import FeatureMask, MaskedFeatures

class CubeDataset(Dataset):
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
        self.non_informative_feature_variance = non_informative_feature_variance

        self._informative_feature_std = torch.sqrt(informative_feature_variance)
        self._non_informative_feature_std = torch.sqrt(
            self.non_informative_feature_variance
        )

    def generate_data(self) -> None:
        rng = torch.Generator()
        rng.manual_seed(self.seed)
        # Each coordinate is drawn from a Bernoulli distribution with p=0.5, which is the same as uniform
        coords = torch.randint(
            low=0, high=2, size=(self.data_points, 3), dtype=torch.int64, generator=rng
        )
        # Each corner in the cube is a different label
        labels = torch.einsum(
            "bi,i->b", coords, torch.tensor([1, 2, 4], dtype=torch.int64)
        )
        # Coords have noise
        coords = coords.float()
        coords += (
            torch.randn(self.data_points, 3, dtype=torch.float32, generator=rng)
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
    def __init__(
        self,
        n_samples: int = 1000,
        sigma_bin: float = 0.1,
        sigma_cube: float = 1.0,
        bin_feature_cost: float = 5.0,
        n_dummy_features: int = 10,
        seed: int = 123,
        non_informative_feature_mean: float = 0.5,
        non_informative_feature_variance: float = 0.3,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.sigma_bin = sigma_bin
        self.sigma_cube = sigma_cube
        self.bin_feature_cost = bin_feature_cost
        self.n_dummy_features = n_dummy_features
        self.seed = seed
        self.non_info_mean = non_informative_feature_mean
        self.non_info_std = torch.sqrt(non_informative_feature_variance)

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
        self.feature_names = None

        # Generate upon initialization
        self.generate_data()

    def generate_data(self) -> None:
        rng = torch.Generator()
        rng.manual_seed(self.seed)

        # Draw labels and context
        y_int = torch.randint(0, self.n_classes, (self.n_samples,), dtype=torch.int64, generator=rng)
        S = torch.randint(0, self.n_context_groups, (self.n_samples,), dtype=torch.int64, generator=rng)

        # Binary codes for labels (8×3)
        binary_codes = torch.stack([
            torch.tensor([int(b) for b in format(i, '03b')], dtype=torch.float32)
            for i in range(self.n_classes)
        ], dim=0)

        # Initialize feature blocks
        X_context = S.unsqueeze(1).float()
        X_bin = torch.rand(self.n_samples, self.n_bin_features, generator=rng)
        X_cube = torch.rand(self.n_samples, self.n_cube_features, generator=rng)
        X_dummy = torch.normal(
            mean=self.non_info_mean,
            std=self.non_info_std,
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
                mean=mu_bin, std=self.sigma_bin, size=(self.group_size,), generator=rng
            )

            # Cube features: 3 bumps
            idxs = [(lbl + j) % self.n_cube_features for j in range(3)]
            X_cube[i, idxs] = torch.normal(
                mean=mu_bin, std=self.sigma_cube, size=(3,), generator=rng
            )

        # Concatenate all features
        self.features = torch.cat([X_context, X_bin, X_cube, X_dummy], dim=1)

        # Build costs vector
        total_dim = self.features.shape[1]
        costs = torch.ones(total_dim)
        costs[1:1 + self.n_bin_features] = self.bin_feature_cost
        self.costs = costs

        # One-hot labels
        #self.labels = torch.nn.functional.one_hot(y_int, num_classes=self.n_classes).float()
        self.labels = y_int

        # Feature names
        names = ['context']
        names += [f'bin_{i}' for i in range(self.n_bin_features)]
        names += [f'cube_{i}' for i in range(self.n_cube_features)]
        names += [f'dummy_{i}' for i in range(self.n_dummy_features)]
        self.feature_names = names

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
                'feature_names': self.feature_names,
                'config': {
                    'n_samples': self.n_samples,
                    'sigma_bin': self.sigma_bin,
                    'sigma_cube': self.sigma_cube,
                    'bin_feature_cost': self.bin_feature_cost,
                    'n_dummy_features': self.n_dummy_features,
                    'seed': self.seed,
                    'non_informative_feature_mean': self.non_info_mean,
                    'non_informative_feature_variance': self.non_info_std ** 2,
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
        ds.feature_names = data['feature_names']
        return ds






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
