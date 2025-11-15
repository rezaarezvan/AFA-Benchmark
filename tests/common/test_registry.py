from pathlib import Path

from common.registry import (
    AFA_DATASET_PATH_REGISTRY,
    EVALUATION_REGISTRY,
    TRAINING_REGISTRY,
)


def test_registry():
    """Test the registry to ensure that all AFA methods and datasets are registered correctly."""
    # Check that each dataset path listed in AFA_DATASET_PATH_REGISTRY exists
    for dataset_name, paths in AFA_DATASET_PATH_REGISTRY.items():
        for path in paths:
            path = Path(path)
            assert path.exists(), f"Dataset path {path} does not exist."
            assert path.is_file(), f"Dataset path {path} is not a file."
            assert path.suffix == ".pt", (
                f"Dataset path {path} does not have .pt suffix."
            )

    # Check that each trained afa method path listed in TRAINING_REGISTRY exists
    for (
        afa_method_name,
        train_data_path,
    ), model_path in TRAINING_REGISTRY.items():
        model_path = Path(model_path)
        assert model_path.exists(), f"Model path {model_path} does not exist."
        assert model_path.is_file(), f"Model path {model_path} is not a file."
        assert model_path.suffix == ".pt", (
            f"Model path {model_path} does not have .pt suffix."
        )
    assert len(set(TRAINING_REGISTRY.values())) == len(TRAINING_REGISTRY), (
        "Duplicate model paths in TRAINING_REGISTRY. Please check the registry."
    )

    # Check that each evaluation result path listed in EVALUATION_REGISTRY exists
    for (
        model_path,
        val_data_path,
    ), eval_results_path in EVALUATION_REGISTRY.items():
        eval_results_path = Path(eval_results_path)
        assert eval_results_path.exists(), (
            f"Evaluation results path {eval_results_path} does not exist."
        )
        assert eval_results_path.is_file(), (
            f"Evaluation results path {eval_results_path} is not a file."
        )
        assert eval_results_path.suffix == ".pt", (
            f"Evaluation results path {eval_results_path} does not have .pt suffix."
        )
    assert len(set(EVALUATION_REGISTRY.values())) == len(
        EVALUATION_REGISTRY
    ), (
        "Duplicate model paths in EVALUATION_REGISTRY. Please check the registry."
    )
