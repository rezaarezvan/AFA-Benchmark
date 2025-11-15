import os
import torch
import wandb
import random
import numpy as np

from typing import Any
from pathlib import Path
from torch import Tensor, nn
from jaxtyping import Bool, Float
from collections.abc import Generator
from contextlib import contextmanager

from afabench.common.custom_types import AFADataset
from afabench.common.registry import get_afa_dataset_class


def set_seed(seed: int | None) -> int:
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed


def get_class_probabilities(
    labels: Bool[Tensor, "*batch n_classes"],
) -> Float[Tensor, "n_classes"]:
    """Return the class probabilities for a given set of labels."""
    class_counts = labels.float().sum(dim=0)
    class_probabilities = class_counts / class_counts.sum()
    return class_probabilities


def get_folders_with_matching_params(
    folder: Path, mapping: dict[str, Any]
) -> list[Path]:
    """Get all folders in a given folder that have a matching params.yml file."""
    matching_folders = [
        f
        for f in folder.iterdir()
        if yaml_file_matches_mapping(f / "params.yml", mapping)
    ]

    return matching_folders


def load_dataset_artifact(
    artifact_name: str,
) -> tuple[AFADataset, AFADataset, AFADataset, dict[str, Any]]:
    """Load train, validation, and test datasets from a WandB artifact, together with its metadata."""
    dataset_artifact = wandb.use_artifact(artifact_name, type="dataset")
    dataset_artifact_dir = Path(dataset_artifact.download())
    # The dataset dir should contain the files train.pt, val.pt and test.pt
    artifact_filenames = [f.name for f in dataset_artifact_dir.iterdir()]
    assert {"train.pt", "val.pt", "test.pt"}.issubset(
        artifact_filenames
    ), f"Dataset artifact must contain train.pt, val.pt and test.pt files. Instead found: {
        artifact_filenames
    }"

    dataset_class = get_afa_dataset_class(
        dataset_artifact.metadata["dataset_type"]
    )
    train_dataset: AFADataset = dataset_class.load(
        dataset_artifact_dir / "train.pt"
    )
    val_dataset: AFADataset = dataset_class.load(
        dataset_artifact_dir / "val.pt"
    )
    test_dataset: AFADataset = dataset_class.load(
        dataset_artifact_dir / "test.pt"
    )

    return train_dataset, val_dataset, test_dataset, dataset_artifact.metadata


@contextmanager
def eval_mode(*models: nn.Module) -> Generator[None, None, None]:
    was_training = [model.training for model in models]
    try:
        for model in models:
            model.eval()
        yield
    finally:
        for model, mode in zip(models, was_training, strict=False):
            model.train(mode)


def dict_with_prefix(prefix: str, d: dict[str, Any]) -> dict[str, Any]:
    """Return a dictionary with all keys prefixed by ."""
    return {f"{prefix}{k}": v for k, v in d.items()}
