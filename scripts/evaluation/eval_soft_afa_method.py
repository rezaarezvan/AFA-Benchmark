import re
import wandb
import hydra
import torch
import logging

from typing import Any
from pathlib import Path
from omegaconf import OmegaConf
from tempfile import NamedTemporaryFile

from afabench.common.config_classes import SoftEvalConfig
from afabench.common.utils import load_dataset_artifact, set_seed
from afabench.eval.soft_budget import eval_soft_budget_afa_method
from afabench.common.registry import (
    get_afa_classifier_class,
    get_afa_method_class,
)

from afabench.common.afa_uncoverings import (
    one_based_index_uncover_fn,
    get_image_patch_uncover_fn,
)

from afabench.common.custom_types import (
    AFAClassifier,
    AFADataset,
    AFAMethod,
)


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
        f"Loading trained AFA method of class {
            method_class.__name__
        } from artifact {artifact_name}"
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
        f"Loading trained classifier of class {
            classifier_class_name
        } from artifact {artifact_name}"
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
        f"The trained method artifact {
            trained_method_artifact_name
        } and the trained classifier artifact {
            trained_classifier_artifact_name
        } "
        f"should have been trained on the same dataset, but they are not. Trained method was trained on {
            method_artifact.metadata['dataset_artifact_name']
        }, classifier was trained on {
            classifier_run.config['dataset_artifact_name']
        }."
    )

    log.debug(
        f"Method and classifier artifacts are compatible and trained on the same dataset: {
            method_artifact.metadata['dataset_artifact_name']
        }"
    )


log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/soft_eval",
    config_name="config",
)
def main(cfg: SoftEvalConfig) -> None:
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")

    run = wandb.init(
        job_type="evaluation",
        config=OmegaConf.to_container(cfg, resolve=True),  # pyright: ignore[reportArgumentType]
        dir="extra/wandb",
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
        msg = f"cfg.dataset_split should either be 'validation' or 'testing', not {
            cfg.dataset_split
        }"
        raise ValueError(msg)
    log.info("Loaded trained AFA method and dataset from artifacts")

    # Load a classifier if it was specified
    external_afa_predict_fn = None
    if cfg.trained_classifier_artifact_name:
        if cfg.validate_artifacts:
            validate_artifacts(
                cfg.trained_method_artifact_name,
                cfg.trained_classifier_artifact_name,
            )
        external_afa_predict_fn, classifier_metadata = (
            load_trained_classifier_artifact(
                cfg.trained_classifier_artifact_name,
                device=torch.device(cfg.device),
            )
        )
        log.info("Loaded external classifier")
    else:
        log.info(
            "No external classifier provided; using builtin predictor only."
        )

    # Some methods need to have the cost parameter set during evaluation
    if hasattr(afa_method, "set_cost_param"):
        assert cfg.cost_param is not None, (
            "cfg.cost_param should be set for methods that need to set the cost parameter during evaluation"
        )
        afa_method.set_cost_param(cfg.cost_param)

    # Do the evaluation
    log.info(
        f"Starting evaluation with soft budget, batch size {cfg.batch_size}"
    )
    modality = getattr(afa_method, "modality", "tabular")
    # is_image = modality == "image"
    image_mask_width = getattr(afa_method, "mask_width", None)
    image_patch_size = getattr(afa_method, "patch_size", 1)
    n_patches = getattr(afa_method, "n_patches", 1)
    uncover_fn = None
    if modality == "image":
        x, _ = dataset[0]
        assert x.ndim == 3
        C, H, W = x.shape
        uncover_fn = get_image_patch_uncover_fn(
            image_side_length=H, n_channels=C, patch_size=image_patch_size
        )
    else:
        uncover_fn = one_based_index_uncover_fn

    df_eval = eval_soft_budget_afa_method(
        afa_select_fn=afa_method.select,
        dataset=dataset,
        external_afa_predict_fn=external_afa_predict_fn,
        afa_uncover_fn=uncover_fn,
        builtin_afa_predict_fn=afa_method.predict
        if afa_method.has_builtin_classifier
        else None,
        only_n_samples=cfg.eval_only_n_samples,
        device=torch.device(cfg.device),
        batch_size=cfg.batch_size,
        patch_size=image_patch_size,
    )
    # Add columns to conform to expected format (snake_case)
    df_eval["method"] = method_metadata["method_type"]
    df_eval["training_seed"] = method_metadata["seed"]
    cost_param = afa_method.cost_param
    assert cost_param is not None, (
        "Cost parameter should not be None for soft budget methods"
    )
    df_eval["cost_parameter"] = cost_param
    df_eval["dataset"] = method_metadata["dataset_type"]

    # Log to wandb for debugging purposes
    run.log({"soft_eval_df": wandb.Table(dataframe=df_eval)})

    # Save results as wandb artifact
    trained_base = cfg.trained_method_artifact_name.split(":")[0]
    trained_base_cost = re.sub(
        r"-budget_[^-]+", f"-costparam_{cfg.cost_param}", trained_base
    )
    eval_results_artifact = wandb.Artifact(
        name=f"{trained_base_cost}-{
            cfg.trained_classifier_artifact_name.split(':')[0]
            if cfg.trained_classifier_artifact_name
            else 'builtin'
        }",
        type="soft_eval_results",
        metadata={
            "dataset_type": method_metadata["dataset_type"],
            "method_type": method_metadata["method_type"],
            "seed": method_metadata["seed"],
        },
    )
    with NamedTemporaryFile("w", delete=False) as f:
        df_save_path = Path(f.name)
        df_eval.to_csv(df_save_path, index=False)
    eval_results_artifact.add_file(
        str(df_save_path), name="soft_eval_data.csv"
    )
    run.log_artifact(
        eval_results_artifact, aliases=cfg.output_artifact_aliases
    )
    run.finish()


if __name__ == "__main__":
    main()
