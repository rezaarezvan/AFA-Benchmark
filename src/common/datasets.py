import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import Dataset

from common.custom_types import FeatureMask, MaskedFeatures


class CubeDataset(Dataset):
    """
    The Cube dataset, as described in the paper "Minimizing data consumption with sequential online feature selection" (https://doi.org/10.1007/s13042-012-0092-x).

    Implements the AFADataset protocol.
    """

    def __init__(
        self,
        n_features: int = 20,
        data_points: int = 20000,
        sigma: float = 0.1,
        seed: int = 123,
    ):
        super().__init__()
        self.n_features = n_features
        self.data_points = data_points
        self.sigma = sigma
        self.seed = seed

    def generate_data(self) -> None:
        rng = torch.Generator()
        rng.manual_seed(self.seed)
        # Each coordinate is drawn from a Bernoulli distribution with p=0.5, which is the same as uniform
        coords: Float[Tensor, "{data_points} 3"] = torch.randint(
            low=0, high=2, size=(self.data_points, 3), dtype=torch.int64, generator=rng
        )
        # Each corner in the cube is a different label
        labels = torch.einsum(
            "bi,i->b", coords, torch.tensor([1, 2, 4], dtype=torch.int64)
        )
        # Add Gaussian noise to coords
        coords = coords.float()
        coords += (
            torch.randn(self.data_points, 3, dtype=torch.float32, generator=rng)
            * self.sigma
        )
        # The final features are the coordinates offset according to the labels, and uniform noise for all other features
        self.features = torch.zeros(
            self.data_points, self.n_features, dtype=torch.float32
        )
        for i in range(self.data_points):
            offset: int = labels[i].item()
            self.features[i, offset : offset + 3] += coords[i]
            # uniform noise on all other features
            self.features[i, :offset] = torch.rand(
                (1, offset), dtype=torch.float32, generator=rng
            )
            self.features[i, offset + 3 :] = torch.rand(
                (1, self.n_features - offset - 3), dtype=torch.float32, generator=rng
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
                    "sigma": self.sigma,
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
