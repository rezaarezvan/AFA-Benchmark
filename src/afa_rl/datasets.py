import lightning as pl
import torch
from jaxtyping import Shaped
from torch import Tensor
from torch.utils.data import DataLoader

from afa_rl.custom_types import AFADatasetFn
from common.custom_types import Features, Label


def get_wrapped_batch(
    t: Shaped[Tensor, "batch *rem"], idx: int, numel: int
) -> Shaped[Tensor, "{num_elems} *rem"]:
    """
    Get a batch of size num_elems from a tensor t, starting at index idx, wrapping around if necessary.
    """
    n = len(t)
    repeated = t.repeat((numel // n) + 2, *[1] * (t.ndim - 1))
    return repeated[idx : idx + numel]


def get_afa_dataset_fn(features, labels) -> AFADatasetFn:
    """
    Given features and labels, return a function that can be used to get batches of AFA data.
    """
    idx = 0  # keep track of where in the dataset we are

    def afa_dataset_fn(
        batch_size: torch.Size, move_on: bool = True
    ) -> tuple[Features, Label]:
        nonlocal idx, features, labels
        local_features = get_wrapped_batch(features, idx, batch_size.numel())
        local_labels = get_wrapped_batch(labels, idx, batch_size.numel())
        if move_on:
            idx = (idx + batch_size.numel()) % len(features)
        local_features = local_features.reshape(*batch_size, local_features.shape[-1])
        local_labels = local_labels.reshape(*batch_size, local_labels.shape[-1])
        return local_features, local_labels

    return afa_dataset_fn


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
