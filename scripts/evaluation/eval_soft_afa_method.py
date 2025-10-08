"""Evaluate a single AFA method with soft budget on a dataset, using a trained classifier if specified."""

import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from common.config_classes import SoftEvalConfig
from eval.metrics import eval_soft_budget_afa_method
import hydra
import torch
from omegaconf import OmegaConf

import wandb
from common.custom_types import (
    AFAClassifier,
    AFADataset,
    AFAMethod,
    AFAPredictFn,
)
from common.registry import get_afa_classifier_class, get_afa_method_class
from common.utils import load_dataset_artifact, set_seed


def load_trained_method_artifacts(
    artifact_name: str, device: torch.device | None = None
) -> tuple[
    AFADataset,
    AFADataset,
    AFADataset,
    AFAMethod,
    dict[str, Any],  # method metadata
]:
    """Load a trained afa method and the dataset it was trained on, from a WandB artifact."""
    if device is None:
        device = torch.device("cpu")
    trained_method_artifact = wandb.use_artifact(
        artifact_name, type="trained_method"
    )
    trained_method_artifact_dir = Path(trained_method_artifact.download())
    method_class = get_afa_method_class(
        trained_method_artifact.metadata["method_type"]
    )
    log.debug(
        f"Loading trained AFA method of class {method_class.__name__} from artifact {artifact_name}"
    )
    method = method_class.load(trained_method_artifact_dir, device=device)

    # Load the dataset that the method was trained on
    train_dataset, val_dataset, test_dataset, _ = load_dataset_artifact(
        trained_method_artifact.metadata["dataset_artifact_name"]
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        method,
        trained_method_artifact.metadata,
    )


def load_trained_classifier_artifact(
    artifact_name: str, device: torch.device | None = None
) -> tuple[AFAClassifier, dict[str, Any]]:  # classifier metadata
    """Load a trained masked classifier from a WandB artifact."""
    if device is None:
        device = torch.device("cpu")

    trained_classifier_artifact = wandb.use_artifact(
        artifact_name, type="trained_classifier"
    )
    trained_classifier_artifact_dir = Path(
        trained_classifier_artifact.download()
    )
    classifier_class_name = trained_classifier_artifact.metadata[
        "classifier_class_name"
    ]
    classifier_class = get_afa_classifier_class(classifier_class_name)
    log.debug(
        f"Loading trained classifier of class {classifier_class_name} from artifact {artifact_name}"
    )
    classifier = classifier_class.load(
        trained_classifier_artifact_dir / "classifier.pt", device=device
    )

    return classifier, trained_classifier_artifact.metadata


def validate_artifacts(
    trained_method_artifact_name: str,
    trained_classifier_artifact_name: str,
) -> None:
    """
    Validate that the trained method and classifier artifacts are compatible.

    They should have been trained on the same dataset.
    """
    method_artifact = wandb.use_artifact(
        trained_method_artifact_name, type="trained_method"
    )
    classifier_artifact = wandb.use_artifact(
        trained_classifier_artifact_name, type="trained_classifier"
    )

    classifier_run = classifier_artifact.logged_by()
    assert classifier_run is not None

    assert (
        method_artifact.metadata["dataset_artifact_name"]
        == classifier_run.config["dataset_artifact_name"]
    ), (
        f"The trained method artifact {trained_method_artifact_name} and the trained classifier artifact {trained_classifier_artifact_name} "
        "should have been trained on the same dataset, but they are not."
    )

    log.debug(
        f"Method and classifier artifacts are compatible and trained on the same dataset: {method_artifact.metadata['dataset_artifact_name']}"
    )


log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../../conf/soft_eval", config_name="config"
)
def main(cfg: SoftEvalConfig) -> None:
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")

    run = wandb.init(
        job_type="evaluation",
        config=OmegaConf.to_container(cfg, resolve=True),  # pyright: ignore[reportArgumentType]
        dir="wandb",
    )

    # Log W&B run URL
    log.info(f"W&B run initialized: {run.name} ({run.id})")
    log.info(f"W&B run URL: {run.url}")

    # Load trained afa method and dataset from artifacts
    _train_dataset_, val_dataset, test_dataset, afa_method, method_metadata = (
        load_trained_method_artifacts(
            cfg.trained_method_artifact_name, device=torch.device(cfg.device)
        )
    )
    if cfg.dataset_split == "validation":
        dataset = val_dataset
    elif cfg.dataset_split == "testing":
        dataset = test_dataset
    else:
        msg = f"cfg.dataset_split should either be 'validation' or 'testing', not {cfg.dataset_split}"
        raise ValueError(msg)
    log.info("Loaded trained AFA method and dataset from artifacts")

    # Load a classifier if it was specified
    if cfg.trained_classifier_artifact_name:
        if cfg.validate_artifacts:
            validate_artifacts(
                cfg.trained_method_artifact_name,
                cfg.trained_classifier_artifact_name,
            )
        afa_predict_fn, classifier_metadata = load_trained_classifier_artifact(
            cfg.trained_classifier_artifact_name,
            device=torch.device(cfg.device),
        )
        classifier_type = classifier_metadata["classifier_type"]
        log.info("Loaded external classifier")
    else:
        log.info("Using builtin classifier")
        afa_predict_fn: AFAPredictFn = afa_method.predict
        if method_metadata["method_type"] == "aaco":
            classifier_type = "MaskedMLPClassifier"
        else:
            classifier_type = "builtin"

    # Hard budget is ignored during evaluation of soft budget methods

    # Do the evaluation
    log.info("Starting evaluation with soft budget")
    df = eval_soft_budget_afa_method(
        afa_select_fn=afa_method.select,
        dataset=dataset,
        external_afa_predict_fn=afa_predict_fn,
        builtin_afa_predict_fn=afa_method.predict
        if afa_method.has_builtin_classifier
        else None,
        only_n_samples=cfg.eval_only_n_samples,
        device=torch.device(cfg.device),
    )
    # Add columns to conform to expected format (snake_case)
    df["method"] = method_metadata["method_type"]
    df["training_seed"] = method_metadata["seed"]
    cost_param = afa_method.cost_param
    assert cost_param is not None, (
        "Cost parameter should not be None for soft budget methods"
    )
    df["cost_parameter"] = cost_param
    df["dataset"] = method_metadata["dataset_type"]

    # Log to wandb for debugging purposes
    run.log({"soft_eval_df": wandb.Table(dataframe=df)})

    # Save results as wandb artifact
    eval_results_artifact = wandb.Artifact(
        name=f"{cfg.trained_method_artifact_name.split(':')[0]}-soft-{cfg.trained_classifier_artifact_name.split(':')[0] if cfg.trained_classifier_artifact_name else 'builtin'}",
        type="soft_eval_results",
        metadata={
            "dataset_type": method_metadata["dataset_type"],
            "method_type": method_metadata["method_type"],
            "seed": method_metadata["seed"],
            "classifier_type": classifier_type,
        },
    )
    with NamedTemporaryFile("w", delete=False) as f:
        df_save_path = Path(f.name)
        df.to_csv(df_save_path, index=False)
    eval_results_artifact.add_file(str(df_save_path), name="soft_eval.csv")
    run.log_artifact(
        eval_results_artifact, aliases=cfg.output_artifact_aliases
    )
    run.finish()


if __name__ == "__main__":
    main()
