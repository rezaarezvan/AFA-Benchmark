import json
import logging
from pathlib import Path
from typing import Any, cast

import hydra
import torch
import wandb
from omegaconf import OmegaConf

from afabench.common.config_classes import EvalConfig
from afabench.common.utils import (
    set_seed,
)
from afabench.eval.eval import eval_afa_method

log = logging.getLogger(__name__)


def load(
    method_artifact_path: Path,
    unmasker_artifact_path: Path,
    initializer_artifact_path: Path,
    dataset_artifact_path: Path,
    dataset_split: str,
    classifier_artifact_path: Path | None = None,
    device: torch.device | None = None,
) -> tuple[
    AFAMethod, AFAUnmasker, AFAInitializer, AFADataset, AFAClassifier | None
]:
    # Load method
    method = load_method_artifact(method_artifact_path)
    log.info(f"Loaded AFA method from {method_artifact_path}")

    # Load unmasker
    unmasker = load_unmasker_artifact(unmasker_artifact_path)
    log.info(f"Loaded unmasker from {unmasker_artifact_path}")

    # Load initializer
    initializer = load_initializer_artifact(initializer_artifact_path)
    log.info(f"Loaded initializer from {initializer_artifact_path}")

    # Load dataset
    dataset = load_dataset_artifact(dataset_artifact_path, dataset_split)
    log.info(f"Loaded dataset from {dataset_artifact_path}")

    # Load external classifier if specified
    if classifier_artifact_path is not None:
        device = torch.device("cpu") if device is None else device
        classifier = load_classifier_artifact(
            classifier_artifact_path, device=device
        )
        log.info(
            f"Loaded external classifier from {classifier_artifact_path}."
        )
    else:
        classifier = None
        log.info("No external classifier provided; using builtin classifier.")

    return (
        method,
        unmasker,
        initializer,
        dataset,
        classifier,
    )


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/soft_eval",
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
    ) = load(
        method_artifact_path=Path(cfg.method_artifact_path),
        unmasker_artifact_path=Path(cfg.unmasker_artifact_path),
        initializer_artifact_path=Path(cfg.initializer_artifact_path),
        dataset_artifact_path=Path(cfg.dataset_artifact_path),
        dataset_split=cfg.dataset_split,
        classifier_artifact_path=(
            Path(cfg.classifier_artifact_path)
            if cfg.classifier_artifact_path is not None
            else None
        ),
        device=torch.device(cfg.device),
    )

    if cfg.hard_budget is not None:
        hard_budget_str = f"hard budget {cfg.hard_budget}"
    else:
        hard_budget_str = "no hard budget"
    log.info(
        f"Starting evaluation with batch size {cfg.batch_size} and {hard_budget_str}."
    )

    # Do the evaluation
    df_eval = eval_afa_method(
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

    # Remove "train_" prefix from method_artifact_name
    eval_artifact_name = method_artifact_name.replace("train_", "")

    # Parse method name (first part before underscore)
    method_name = eval_artifact_name.split("_")[0]

    # Create eval directory: extra/result/{method_name}/eval/{artifact_name}/
    base_dir = Path("extra/result")
    # remove "methodname_" prefix from eval_artifact_name
    eval_artifact_name = eval_artifact_name.split("_", 1)[-1]
    eval_dir = base_dir / method_name / "eval" / eval_artifact_name
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV directly
    csv_path = eval_dir / "soft_eval_data.csv"
    df_eval.to_csv(csv_path, index=False)
    log.info(f"Saved evaluation data to CSV at: {csv_path}")

    # Save metadata
    eval_metadata = {
        "dataset_type": method_metadata["dataset_type"],
        "method_type": method_metadata["method_type"],
        "seed": method_metadata["seed"],
        "cost_param": cfg.cost_param,
        "eval_type": "soft_budget",
        "dataset_split": cfg.dataset_split,
        "classifier_artifact_name": cfg.trained_classifier_artifact_name,
    }

    with open(eval_dir / "metadata.json", "w") as f:
        json.dump(eval_metadata, f, indent=2)

    log.info(f"Evaluation results saved to: {eval_dir}")

    if run:
        run.finish()


if __name__ == "__main__":
    main()
