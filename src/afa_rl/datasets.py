import lightning as pl
import torch
from jaxtyping import Float, Shaped
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from afa_rl.custom_types import DatasetFn, Sample


def get_wrapped_batch(
    t: Shaped[Tensor, "batch *rem"], idx: int, num_elems: int
) -> Shaped[Tensor, "{num_elems} *rem"]:
    """
    Get a batch of size num_elems from a tensor t, starting at index idx, wrapping around if necessary.
    """
    n = len(t)
    repeated = t.repeat((num_elems // n) + 2, *[1] * (t.ndim - 1))
    return repeated[idx : idx + num_elems]


def get_dataset_fn(features, labels) -> DatasetFn:
    """
    Given a dataset, return a function that can be used to get batches from it.
    """
    idx = 0  # keep track of where in the dataset we are

    def data_fn(batch_size: torch.Size, move_on: bool = True) -> Sample:
        nonlocal idx, features, labels
        num_elems = batch_size.numel()
        local_features = get_wrapped_batch(features, idx, num_elems)
        local_labels = get_wrapped_batch(labels, idx, num_elems)
        if move_on:
            idx = (idx + num_elems) % len(features)
        return Sample(local_features, local_labels)

    return data_fn


class CubeDataset(Dataset):
    """
    The Cube dataset, as described in the paper "Minimizing data consumption with sequential online feature selection" (https://doi.org/10.1007/s13042-012-0092-x).
    """

    def __init__(
        self,
        n_features: int = 20,
        data_points: int = 20000,
        sigma: float = 0.1,
        seed: int = 123,
    ):
        rng = torch.Generator()
        rng.manual_seed(seed)
        # Each coordinate is drawn from a Bernoulli distribution with p=0.5, which is the same as uniform
        coords: Float[Tensor, "{data_points} 3"] = torch.randint(
            low=0, high=2, size=(data_points, 3), dtype=torch.int64, generator=rng
        )
        # Each corner in the cube is a different label
        labels = torch.einsum(
            "bi,i->b", coords, torch.tensor([1, 2, 4], dtype=torch.int64)
        )
        # Add Gaussian noise to coords
        coords = coords.float()
        coords += (
            torch.randn(data_points, 3, dtype=torch.float32, generator=rng) * sigma
        )
        # The final features are the coordinates offset according to the labels, and uniform noise for all other features
        self.features = torch.zeros(data_points, n_features, dtype=torch.float32)
        for i in range(data_points):
            offset: int = labels[i].item()
            self.features[i, offset : offset + 3] += coords[i]
            # uniform noise on all other features
            self.features[i, :offset] = torch.rand(
                (1, offset), dtype=torch.float32, generator=rng
            )
            self.features[i, offset + 3 :] = torch.rand(
                (1, n_features - offset - 3), dtype=torch.float32, generator=rng
            )
        # Convert labels to one-hot encoding
        self.labels = torch.nn.functional.one_hot(labels, num_classes=8).float()

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def __len__(self):
        return len(self.features)


class DataModuleFromDataset(pl.LightningDataModule):
    def __init__(self, dataset, batch_size=32, train_ratio=0.8, num_workers=1):
        # TODO: does not work with num_workers > 1
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        train_size = int(self.train_ratio * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
