import gc
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import hydra
import torch
import wandb
from omegaconf import OmegaConf

from afabench.common.afa_methods import RandomDummyAFAMethod
from afabench.common.config_classes import (
    RandomDummyTrainConfig,
)
from afabench.common.registry import get_afa_dataset_class
from afabench.common.utils import save_artifact, set_seed

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/train/randomdummy",
    config_name="config",
)
def main(cfg: RandomDummyTrainConfig) -> None:
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")

    if cfg.use_wandb:
        run = wandb.init(
            config=cast(
                "dict[str, Any]", OmegaConf.to_container(cfg, resolve=True)
            ),
            job_type="training",
            tags=["randomdummy"],
            dir="extra/wandb",
        )
        # Log W&B run URL
        log.info(f"W&B run initialized: {run.name} ({run.id})")
        log.info(f"W&B run URL: {run.url}")
    else:
        run = None

    # HACK: Load dataset
    with (Path(cfg.dataset_artifact_path) / "metadata.json").open("r") as f:
        dataset_metadata = json.load(f)
    dataset_type = dataset_metadata["dataset_type"]
    dataset_class = get_afa_dataset_class(dataset_type)
    train_dataset = dataset_class.load(
        Path(cfg.dataset_artifact_path) / "train.pt"
    )

    # dataset_type = dataset_metadata["dataset_type"]
    # split = dataset_metadata["split_idx"]
    assert len(train_dataset.feature_shape) == 1, "Only 1D features supported"
    assert len(train_dataset.label_shape) == 1, "Only 1D labels supported"
    n_features = train_dataset.feature_shape[0]
    n_classes = train_dataset.label_shape[-1]

    log.info(f"Dataset: {dataset_type}")
    log.info(f"Features: {n_features}, Classes: {n_classes}")
    log.info(f"Training samples: {len(train_dataset)}")

    # Get cost parameter
    cost = (
        cfg.cost_param
        if cfg.cost_param is not None
        else cfg.aco.acquisition_cost
    )

    afa_method = RandomDummyAFAMethod(
        device=torch.device("cpu"),
        n_classes=n_classes,
        prob_select_0=cfg.cost_param,
    )

    # Save the method to a temporary directory and load it again to ensure it is saved correctly
    with TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        afa_method.save(temp_path)

        # Prepare metadata
        metadata = {
            "method_type": "RandomDummy",
            "dataset_type": dataset_type,
            "dataset_artifact_name": cfg.dataset_artifact_path,
            "budget": None,
            "seed": cfg.seed,
            "cost_param": cost,
            "split_idx": split,
        }

        # Save artifact to filesystem
        save_artifact(
            artifact_dir=artifact_dir,
            files={f.name: f for f in temp_path.iterdir() if f.is_file()},
            metadata=metadata,
        )

        log.info(f"RandomDummy method saved to {artifact_dir}")

    run.finish()

    gc.collect()  # Force Python GC
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Release cached memory held by PyTorch CUDA allocator
        torch.cuda.synchronize()  # Optional, wait for CUDA ops to finish


if __name__ == "__main__":
    main()
