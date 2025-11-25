import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch

from afabench.common.registry import get_afa_dataset_class

DATASETS_TO_TEST = [
    # (dataset_name, kwargs)
    ("AFAContext", {"n_samples": 10, "seed": 42}),
    ("cube", {"n_samples": 10, "seed": 42}),
    ("diabetes", {"root": "extra/data/misc/diabetes.csv"}),
    ("miniboone", {"root": "extra/data/misc/miniboone.csv"}),
    ("physionet", {"root": "extra/data/misc/physionet.csv"}),
    (
        "imagenette",
        {
            "data_root": "extra/data/",
            "variant_dir": "imagenette2-320",
            "load_subdirs": ("val",),
            "image_size": 224,
            "split_role": "val",
        },
    ),
    ("bank_marketing", {"path": "extra/data/misc/bank_marketing.csv"}),
    ("ckd", {"path": "extra/data/misc/ckd.csv"}),
    ("actg", {"path": "extra/data/misc/actg175.csv"}),
]


@pytest.mark.parametrize(("dataset_name", "kwargs"), DATASETS_TO_TEST)
def test_save_load_dataset_roundtrip(
    dataset_name: str, kwargs: dict[str, Any]
) -> None:
    dataset_class = get_afa_dataset_class(dataset_name)

    # Instantiate dataset
    dataset = dataset_class(**kwargs)
    features, labels = dataset.get_all_data()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "data.pt"
        dataset.save(save_path)
        loaded = dataset_class.load(save_path)
        loaded_features, loaded_labels = loaded.get_all_data()

        assert torch.allclose(features, loaded_features), (
            "Features mismatch after save/load"
        )
        assert torch.allclose(labels, loaded_labels), (
            "Labels mismatch after save/load"
        )
