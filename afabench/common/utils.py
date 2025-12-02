import json
import os
import random
import shutil
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import torch
from torch import nn

from afabench.common.config_classes import InitializerConfig, UnmaskerConfig
from afabench.common.custom_types import (
    AFAClassifier,
    AFADataset,
    AFAInitializer,
    AFAMethod,
    AFAUnmasker,
)
from afabench.common.initializers.utils import get_afa_initializer_from_config
from afabench.common.registry import (
    get_afa_classifier_class,
    get_afa_dataset_class,
    get_afa_method_class,
)
from afabench.common.unmaskers.utils import get_afa_unmasker_from_config


def set_seed(seed: int | None) -> int:
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed


def get_class_frequencies(labels: torch.Tensor) -> torch.Tensor:
    """Return class frequencies for labels of shape (*batch_size, n_classes)."""
    assert labels.shape[-1] > 1, f"Expected multi-class labels, got {
        labels.shape
    }"
    class_counts = labels.flatten(0, -2).float().sum(dim=0)
    return class_counts / class_counts.sum()


@contextmanager
def eval_mode(*models: nn.Module) -> Generator[None, None, None]:
    was_training = [m.training for m in models]
    try:
        for m in models:
            m.eval()
        yield
    finally:
        for m, mode in zip(models, was_training, strict=False):
            m.train(mode)


def save_method_artifact(
    method: AFAMethod,
    save_path: Path,
    metadata: dict[str, Any],
) -> None:
    """Save an AFA method with metadata."""
    if isinstance(save_path, str):
        save_path = Path(save_path)

    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        method.save(temp_path)
        save_artifact(
            artifact_dir=save_path,
            files={f.name: f for f in temp_path.iterdir() if f.is_file()},
            metadata=metadata,
        )


def save_artifact(
    artifact_dir: Path,
    files: dict[str, Path],
    metadata: dict[str, Any],
) -> None:
    """Save artifact files and metadata.json to directory."""
    artifact_dir.mkdir(parents=True, exist_ok=True)

    with Path(artifact_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    for dest_name, source_path in files.items():
        dest = artifact_dir / dest_name
        if source_path.is_file():
            shutil.copy2(source_path, dest)
        elif source_path.is_dir():
            shutil.copytree(source_path, dest, dirs_exist_ok=True)


def load_artifact_metadata(artifact_dir: Path) -> dict[str, Any]:
    """Load metadata.json from artifact directory."""
    path = artifact_dir / "metadata.json"
    if not path.exists():
        msg = f"No metadata.json in {artifact_dir}"
        raise FileNotFoundError(msg)
    with Path(path).open() as f:
        return json.load(f)


def load_pretrained_model(
    path: Path,
    device: torch.device | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load pretrained model checkpoint. Returns (model_path, metadata)."""
    device = device or torch.device("cpu")
    metadata = load_artifact_metadata(path)
    model_path = path / "model.pt"
    if not model_path.exists():
        msg = f"No model.pt in {path}"
        raise FileNotFoundError(msg)
    return model_path, metadata


def load_method_artifact(
    method_artifact_path: Path,
    device: torch.device | None = None,
) -> tuple[AFAMethod, dict[str, Any]]:
    """Load trained AFA method from artifact path."""
    device = device or torch.device("cpu")
    metadata = load_artifact_metadata(method_artifact_path)
    method_class = get_afa_method_class(metadata["method_class_name"])
    return method_class.load(method_artifact_path, device=device), metadata


def load_dataset_artifact(
    dataset_artifact_path: Path,
    split: str,
) -> tuple[AFADataset, dict[str, Any]]:
    """Load single dataset split (train/val/test) from artifact path."""
    if isinstance(dataset_artifact_path, str):
        dataset_artifact_path = Path(dataset_artifact_path)

    if split not in {"train", "val", "test"}:
        msg = f"Invalid split: {split}"
        raise ValueError(msg)
    metadata = load_artifact_metadata(dataset_artifact_path)
    dataset_class = get_afa_dataset_class(metadata["class_name"])
    return dataset_class.load(dataset_artifact_path / f"{split}.pt"), metadata


def load_dataset_splits(
    dataset_artifact_path: Path,
) -> tuple[AFADataset, AFADataset, AFADataset, dict[str, Any]]:
    """Load all dataset splits. Returns (train, val, test, metadata)."""
    for f in ["train.pt", "val.pt", "test.pt", "metadata.json"]:
        if not (dataset_artifact_path / f).exists():
            msg = f"Missing {f} in {dataset_artifact_path}"
            raise FileNotFoundError(msg)
    metadata = load_artifact_metadata(dataset_artifact_path)
    cls = get_afa_dataset_class(metadata["class_name"])

    return (
        cls.load(dataset_artifact_path / "train.pt"),
        cls.load(dataset_artifact_path / "val.pt"),
        cls.load(dataset_artifact_path / "test.pt"),
        metadata,
    )


def load_classifier_artifact(
    classifier_artifact_path: Path,
    device: torch.device | None = None,
) -> tuple[AFAClassifier, dict[str, Any]]:
    """Load trained classifier from artifact path."""
    device = device or torch.device("cpu")
    metadata = load_artifact_metadata(classifier_artifact_path)
    classifier_class = get_afa_classifier_class(metadata["class_name"])
    return classifier_class.load(
        classifier_artifact_path / "classifier.pt", device=device
    ), metadata


def load_eval_components(
    method_artifact_path: Path,
    unmasker_cfg: UnmaskerConfig,
    initializer_cfg: InitializerConfig,
    dataset_artifact_path: Path,
    dataset_split: str,
    classifier_artifact_path: Path,
    device: torch.device | None = None,
) -> tuple[AFAMethod, AFAUnmasker, AFAInitializer, AFADataset, AFAClassifier]:
    """Load all components for evaluation."""
    device = device or torch.device("cpu")

    method, _ = load_method_artifact(method_artifact_path, device=device)
    unmasker = get_afa_unmasker_from_config(unmasker_cfg)
    initializer = get_afa_initializer_from_config(initializer_cfg)
    dataset, _ = load_dataset_artifact(
        dataset_artifact_path=dataset_artifact_path, split=dataset_split
    )
    classifier, _ = load_classifier_artifact(
        classifier_artifact_path=classifier_artifact_path, device=device
    )
    return (method, unmasker, initializer, dataset, classifier)
