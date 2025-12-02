import logging
from pathlib import Path
from typing import Any, cast

import hydra
import torch
import wandb
from omegaconf import OmegaConf

from afabench.afa_oracle.afa_methods import AACOAFAMethod
from afabench.common.config_classes import (
    EvalConfig,
    InitializerConfig,
    UnmaskerConfig,
)
from afabench.common.custom_types import (
    AFAClassifier,
    AFADataset,
    AFAInitializer,
    AFAMethod,
    AFAUnmasker,
)
from afabench.common.initializers.utils import get_afa_initializer_from_config
from afabench.common.unmaskers.utils import get_afa_unmasker_from_config
from afabench.common.utils import (
    load_classifier_artifact,
    load_dataset_artifact,
    load_method_artifact,
    set_seed,
)
from afabench.eval.eval import eval_afa_method

log = logging.getLogger(__name__)


def load(
    method_artifact_path: Path,
    unmasker_cfg: UnmaskerConfig,
    initializer_cfg: InitializerConfig,
    dataset_artifact_path: Path,
    dataset_split: str,
    classifier_artifact_path: Path | None = None,
    device: torch.device | None = None,
) -> tuple[
    AFAMethod,
    AFAUnmasker,
    AFAInitializer,
    AFADataset,
    AFAClassifier | None,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any] | None,
]:
    # Load method
    method, method_metadata = load_method_artifact(method_artifact_path)
    log.info(f"Loaded AFA method from {method_artifact_path}")

    # Load unmasker
    unmasker: AFAUnmasker = get_afa_unmasker_from_config(unmasker_cfg)
    log.info(f"Loaded {unmasker_cfg.class_name}")

    # Load initializer
    initializer: AFAInitializer = get_afa_initializer_from_config(
        initializer_cfg
    )
    log.info(f"Loaded {initializer_cfg.class_name} initializer")

    # Load dataset
    dataset, dataset_metadata = load_dataset_artifact(
        dataset_artifact_path, dataset_split
    )
    log.info(f"Loaded dataset from {dataset_artifact_path}")

    # Load external classifier if specified
    if classifier_artifact_path is not None:
        device = torch.device("cpu") if device is None else device
        classifier, classifier_metadata = load_classifier_artifact(
            classifier_artifact_path, device=device
        )
        log.info(
            f"Loaded external classifier from {classifier_artifact_path}."
        )
    else:
        classifier = None
        classifier_metadata = None
        log.info("No external classifier provided; using builtin classifier.")

    return (
        method,
        unmasker,
        initializer,
        dataset,
        classifier,
        method_metadata,
        dataset_metadata,
        classifier_metadata,
    )


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/eval",
    config_name="config",
)
def main(cfg: EvalConfig) -> None:
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")

    if cfg.use_wandb:
        run = wandb.init(
            job_type="evaluation",
            config=cast(
                "dict[str, Any]",
                OmegaConf.to_container(cfg, resolve=True),
            ),
            dir="extra/wandb",
        )
        log.info(f"W&B run initialized: {run.name} ({run.id})")
        log.info(f"W&B run URL: {run.url}")
    else:
        run = None

    # Load everything
    (
        afa_method,
        unmasker,
        initializer,
        dataset,
        external_classifier,
        method_metadata,
        dataset_metadata,
        external_classifier_metadata,
    ) = load(
        method_artifact_path=Path(cfg.method_artifact_path),
        unmasker_cfg=cfg.unmasker,
        initializer_cfg=cfg.initializer,
        dataset_artifact_path=Path(cfg.dataset_artifact_path),
        dataset_split=cfg.dataset_split,
        classifier_artifact_path=(
            Path(cfg.classifier_artifact_path)
            if cfg.classifier_artifact_path is not None
            else None
        ),
        device=torch.device(cfg.device),
    )

    # We currently only allow validation on the same dataset (but not split) as the method was trained on.
    assert (
        method_metadata["dataset_class_name"] == dataset_metadata["class_name"]
    ), (
        f"Expected dataset class {dataset_metadata['class_name']} to match the dataset used during method training {method_metadata['dataset_class_name']}."
    )

    # Set the seed of everything
    afa_method.set_seed(cfg.seed)
    unmasker.set_seed(cfg.seed)
    initializer.set_seed(cfg.seed)

    if cfg.hard_budget is not None:
        hard_budget_str = f"hard budget {cfg.hard_budget}"
    else:
        hard_budget_str = "no hard budget"
    log.info(
        f"Starting evaluation with batch size {cfg.batch_size} and {
            hard_budget_str
        }."
    )

    if isinstance(afa_method, AACOAFAMethod):
        afa_method.aaco_oracle.set_classifier(
            external_classifier
            if external_classifier is not None
            else afa_method.aaco_oracle.classifier
        )

    afa_method = cast(
        "AFAMethod", afa_method
    )  # TODO: remove once AACOAFAMethod implements AFAMethod
    df_eval = eval_afa_method(
        afa_select_fn=afa_method.select,
        afa_unmask_fn=unmasker.unmask,
        n_selection_choices=unmasker.get_n_selections(
            feature_shape=dataset.feature_shape
        ),
        afa_initialize_fn=initializer.initialize,
        dataset=dataset,
        external_afa_predict_fn=external_classifier.__call__
        if external_classifier is not None
        else None,
        builtin_afa_predict_fn=afa_method.predict
        if afa_method.has_builtin_classifier
        else None,
        only_n_samples=cfg.eval_only_n_samples,
        device=torch.device(cfg.device),
        selection_budget=cfg.hard_budget,
        batch_size=cfg.batch_size,
    )

    # Add metadata columns first
    cost_param = afa_method.cost_param

    # For hard budget evaluation, we don't use the cost parameter
    if cfg.hard_budget is not None:
        train_soft_budget_param = None
        eval_soft_budget_param = None
    else:
        # For soft budget evaluation, use the cost parameter
        assert cost_param is not None, (
            "Cost parameter should not be None for soft budget methods"
        )
        train_soft_budget_param = cost_param
        eval_soft_budget_param = None

    df_eval = df_eval.assign(
        afa_method=method_metadata["method_class_name"],
        train_seed=method_metadata["seed"],
        train_soft_budget_param=train_soft_budget_param,
        dataset=method_metadata["dataset_class_name"],
        selections_performed=df_eval["prev_selections_performed"].apply(len)
        + 1,
        features_observed=df_eval["next_feature_indices"].apply(len),
        eval_seed=cfg.seed,
        train_hard_budget=method_metadata.get("hard_budget"),
        eval_hard_budget=cfg.hard_budget,
        eval_soft_budget_param=eval_soft_budget_param,
    )

    # Identify which classifier columns are present and have non-null values
    classifier_columns = [
        col for col in df_eval.columns if col.endswith("_predicted_class")
    ]

    # Filter to only include classifier columns that have at least some non-null values
    available_classifier_columns = [
        col for col in classifier_columns if bool(df_eval[col].notna().any())
    ]

    if available_classifier_columns:
        # Use pandas melt to pivot longer, but only for available classifiers
        id_vars = [
            col
            for col in df_eval.columns
            if not col.endswith("_predicted_class")
        ]
        df_eval = df_eval.melt(
            id_vars=id_vars,
            value_vars=available_classifier_columns,
            var_name="classifier_type",
            value_name="predicted_class",
        )
        # Clean up the classifier column to remove '_predicted_class' suffix
        df_eval["classifier"] = df_eval["classifier_type"].str.replace(
            "_predicted_class", ""
        )
        df_eval = df_eval.drop("classifier_type", axis=1)
    else:
        # Fallback if no classifier columns found
        df_eval["predicted_class"] = None
        df_eval["classifier"] = "none"

    # Select final columns in the expected order
    df_eval = df_eval[
        [
            "afa_method",
            "classifier",
            "dataset",
            "selections_performed",
            "features_observed",
            "predicted_class",
            "true_class",
            "train_seed",
            "eval_seed",
            "train_hard_budget",
            "eval_hard_budget",
            "train_soft_budget_param",
            "eval_soft_budget_param",
        ]
    ]

    # Save CSV directly
    csv_path = Path(cfg.save_path) / "eval_data.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_eval.to_csv(csv_path, index=False)
    log.info(f"Saved evaluation data to CSV at: {csv_path}")

    # Save metadata
    # TODO:
    # eval_metadata = {
    #     "dataset_class_name": method_metadata["dataset_class_name"],
    #     "method_class_name": method_metadata["method_class_name"],
    #     "seed": method_metadata["seed"],
    #     "cost_param": cost_param,
    #     "eval_type": "soft_budget",
    #     "dataset_split": cfg.dataset_split,
    #     "classifier_artifact_name": cfg.classifier_artifact_path,
    # }
    # Save just metadata (CSV already in place)
    # save_artifact(
    #     artifact_dir=Path(cfg.save_path),
    #     files={},  # No files to copy - CSV already there
    #     metadata=eval_metadata,
    # )
    log.info(f"Evaluation results saved to: {cfg.save_path}")

    if run:
        run.finish()


if __name__ == "__main__":
    main()
