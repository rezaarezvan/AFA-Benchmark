import json
import os
import random
import shutil
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from afabench.common.custom_types import AFAClassifier, AFADataset, AFAMethod
from afabench.common.registry import (
    get_afa_classifier_class,
    get_afa_dataset_class,
    get_afa_method_class,
)


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


def get_class_frequencies(labels: torch.Tensor) -> torch.Tensor:
    """
    Return the class frequencies for a given set of labels.

    Args:
        labels: a (*batch_size, n_classes) boolean tensor.

    Returns:
        A (n_classes,) float tensor of class probabilities.
    """
    assert labels.shape[-1] > 1, (
        f"Expected multi-class labels, got shape {labels.shape}"
    )
    class_counts = labels.flatten(0, -2).float().sum(dim=0)
    class_probabilities = class_counts / class_counts.sum()
    assert class_probabilities.ndim == 1
    return class_probabilities


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


def save_artifact(
    artifact_dir: Path,
    files: dict[str, Path],
    metadata: dict[str, Any],
) -> None:
    """
    Save artifact files and metadata to local filesystem.

    Args:
        artifact_dir: Directory to save artifact (will be created)
        files: Dict mapping destination filename to source file path
        metadata: Metadata dict to save as JSON
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    with open(artifact_dir / "metadata.json", "w") as f:  # noqa: PTH123
        json.dump(metadata, f, indent=2)

    # Copy files
    for dest_name, source_path in files.items():
        if source_path.is_file():
            shutil.copy2(source_path, artifact_dir / dest_name)
        elif source_path.is_dir():
            shutil.copytree(
                source_path, artifact_dir / dest_name, dirs_exist_ok=True
            )

    # Assert files saved
    saved_filenames = [f.name for f in artifact_dir.iterdir()]
    expected_filenames = list(files.keys()) + ["metadata.json"]
    missing = set(expected_filenames) - set(saved_filenames)
    assert not missing, f"Missing files in artifact: {missing}"


def load_artifact_metadata(artifact_dir: Path) -> dict[str, Any]:
    """Load metadata.json from artifact directory."""
    metadata_path = artifact_dir / "metadata.json"
    assert metadata_path.exists(), f"No metadata.json found in {artifact_dir}"

    with open(metadata_path) as f:  # noqa: PTH123
        return json.load(f)


def get_artifact_path(
    artifact_type: str,
    artifact_name: str,
    base_dir: Path,
) -> Path:
    # extra/data/{dataset_name}/{dataset_name}_split_{i}
    if artifact_type == "dataset":
        return base_dir / "data" / artifact_name
    # extra/result/{classifier_name}/
    if artifact_type == "classifier":
        return base_dir / "classifiers" / artifact_name
    # extra/result/{method_name}/train/{artifact_name}
    if artifact_type == "trained_method":
        method_name = artifact_name.split("-")[0]
        return base_dir / method_name / "train" / artifact_name
    # extra/result/{method_name}/pretrain/{artifact_name}
    if artifact_type == "pretrained_model":
        method_name = artifact_name.split("-")[0]
        return base_dir / method_name / "pretrain" / artifact_name
    msg = f"Unknown artifact type: {artifact_type}"
    raise ValueError(msg)


def load_dataset(
    dataset_path: str | Path,
    base_dir: Path,
) -> tuple[AFADataset, AFADataset, AFADataset, dict[str, Any]]:
    """
    Load train, validation, and test datasets from local filesystem.

    Returns: (train_dataset, val_dataset, test_dataset, metadata)
    """
    if isinstance(dataset_path, str):
        dataset_path = Path(dataset_path)

    if not dataset_path.is_absolute():
        dataset_path = get_artifact_path(
            "dataset", str(dataset_path), base_dir
        )

    # Check required files
    required_files = ["train.pt", "val.pt", "test.pt", "metadata.json"]
    artifact_filenames = [
        f.name for f in dataset_path.iterdir() if f.is_file()
    ]

    missing = set(required_files) - set(artifact_filenames)
    if missing:
        msg = f"Dataset at {dataset_path} missing: {missing}. Found: {
            artifact_filenames
        }"
        raise FileNotFoundError(msg)

    # Load metadata and datasets
    metadata = load_artifact_metadata(dataset_path)
    dataset_class = get_afa_dataset_class(metadata["dataset_type"])

    train_dataset: AFADataset = dataset_class.load(dataset_path / "train.pt")
    val_dataset: AFADataset = dataset_class.load(dataset_path / "val.pt")
    test_dataset: AFADataset = dataset_class.load(dataset_path / "test.pt")

    return train_dataset, val_dataset, test_dataset, metadata


def load_classifier(
    classifier_artifact_name: str,
    base_dir: Path,
    device: torch.device | None = None,
) -> AFAClassifier:
    """Load classifier for given dataset and split."""
    if device is None:
        device = torch.device("cpu")

    # masked_mlp_classifier-cube_split_1/
    # of the form "{classifier_type}-{dataset_name}_split_{split}"

    classifier_path = get_artifact_path(
        "classifier", classifier_artifact_name, base_dir
    )

    if not classifier_path.exists():
        msg = f"Classifier not found: {classifier_path}"
        raise FileNotFoundError(msg)

    metadata = load_artifact_metadata(classifier_path)
    classifier_class_name = metadata["classifier_class_name"]

    classifier_class = get_afa_classifier_class(classifier_class_name)
    classifier = classifier_class.load(
        classifier_path / "classifier.pt", device=device
    )

    return classifier


def load_trained_classifier(
    artifact_name: str,
    base_dir: Path,
    device: torch.device | None = None,
) -> tuple[AFAClassifier, dict[str, Any]]:
    """Load trained classifier from filesystem."""
    if device is None:
        device = torch.device("cpu")

    classifier_path = get_artifact_path("classifier", artifact_name, base_dir)

    assert classifier_path.exists(), f"Classifier not found: {classifier_path}"

    metadata = load_artifact_metadata(classifier_path)
    classifier_class_name = metadata["classifier_class_name"]

    classifier_class = get_afa_classifier_class(classifier_class_name)
    classifier = classifier_class.load(
        classifier_path / "classifier.pt", device=device
    )

    return classifier, metadata


def load_trained_method(
    artifact_name: str,
    base_dir: Path,
    device: torch.device | None = None,
) -> tuple[AFADataset, AFADataset, AFADataset, AFAMethod, dict[str, Any]]:
    """Load trained AFA method and dataset."""
    if device is None:
        device = torch.device("cpu")

    # Extract method and stage
    stage = "train"  # default
    if artifact_name.startswith("train_"):
        artifact_name = artifact_name[6:]
        stage = "train"
    elif artifact_name.startswith("eval_"):
        artifact_name = artifact_name[5:]
        stage = "eval"

    # Parse method name (first part before _)
    method_name = artifact_name.split("_")[0]

    # Build search path: extra/result/{method}/{stage}/
    search_dir = base_dir / method_name / stage
    assert search_dir.exists(), f"Method directory not found: {search_dir}"

    # Method search path: extra/result/{method}/{stage}/{artifact_name}
    artifact_name = artifact_name.split(f"{method_name}_", 1)[-1]
    method_path = search_dir / artifact_name

    metadata_path = method_path / "metadata.json"
    assert metadata_path.exists(), f"No metadata.json found in {method_path}"

    metadata = load_artifact_metadata(method_path)
    method_class = get_afa_method_class(metadata["method_type"])
    method = method_class.load(method_path, device=device)

    # Load dataset
    train_dataset, val_dataset, test_dataset, _ = load_dataset(
        metadata["dataset_artifact_name"], base_dir=base_dir
    )

    return train_dataset, val_dataset, test_dataset, method, metadata


def load_pretrained_model(  # noqa: ANN201
    artifact_name: str,
    base_dir: Path,
    device: torch.device | None = None,
):
    """Load pretrained model checkpoint and associated dataset."""
    if device is None:
        device = torch.device("cpu")

    # Strip prefix e.g. "pretrain_covert2023_cube_split_1_seed_42"
    if artifact_name.startswith("pretrain_"):
        artifact_name = artifact_name[len("pretrain_") :]
        stage = "pretrain"
    else:
        stage = "pretrain"

    method_name = artifact_name.split("_")[0]

    # extra/result/<method_name>/pretrain/<artifact_subfolder>
    search_dir = base_dir / method_name / stage
    if not search_dir.exists():
        msg = f"Pretraining folder not found: {search_dir}"
        raise FileNotFoundError(msg)

    artifact_sub = artifact_name.split(f"{method_name}_", 1)[-1]
    method_path = search_dir / artifact_sub

    metadata_path = method_path / "metadata.json"
    if not metadata_path.exists():
        msg = f"No metadata.json found in {method_path}"
        raise FileNotFoundError(msg)

    metadata = load_artifact_metadata(method_path)
    pretrain_cfg = metadata.get("pretrain_config", {})

    model_pt_path = method_path / "model.pt"
    if not model_pt_path.exists():
        msg = f"Missing model.pt at {model_pt_path}"
        raise FileNotFoundError(msg)

    # Load dataset
    train_dataset, val_dataset, test_dataset, dataset_metadata = load_dataset(
        metadata["dataset_artifact_name"], base_dir=base_dir
    )

    return (
        model_pt_path,
        metadata,
        pretrain_cfg,
        train_dataset,
        val_dataset,
        test_dataset,
        dataset_metadata,
    )


def validate_method_classifier_compatibility(
    method_metadata: dict[str, Any],
    classifier_metadata: dict[str, Any],
) -> None:
    """Validate method and classifier were trained on same dataset."""
    method_dataset = method_metadata["dataset_artifact_name"]
    classifier_dataset = classifier_metadata["dataset_artifact_name"]

    # Parse old wandb names
    if "/" in method_dataset:
        method_dataset = method_dataset.split("/")[-1].split(":")[0]
    if "/" in classifier_dataset:
        classifier_dataset = classifier_dataset.split("/")[-1].split(":")[0]

    assert method_dataset == classifier_dataset, (
        f"Method and classifier must be trained on same dataset. "
        f"Method: {method_dataset}, Classifier: {classifier_dataset}"
    )
