import torch
import hydra
import wandb
import logging

from pathlib import Path
from typing import Any, cast
from omegaconf import OmegaConf
from tempfile import TemporaryDirectory
from afa_oracle import create_aaco_method
from common.config_classes import AACOTrainConfig
from common.utils import load_dataset_artifact, set_seed

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../conf/train/aco", config_name="config")
def main(cfg: AACOTrainConfig):
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")
    device = torch.device(cfg.device)

    run = wandb.init(
        config=cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True)),
        job_type="training",
        tags=["aaco"],
        dir="wandb",
    )

    log.info(f"W&B run initialized: {run.name} ({run.id})")
    log.info(f"W&B run URL: {run.url}")

    train_dataset, _, _, dataset_metadata = load_dataset_artifact(
        cfg.dataset_artifact_name
    )

    dataset_type = dataset_metadata["dataset_type"]
    split = dataset_metadata["split_idx"]
    n_features = train_dataset.features.shape[-1]
    n_classes = train_dataset.labels.shape[-1]

    log.info(f"Dataset: {dataset_type}")
    log.info(f"Features: {n_features}, Classes: {n_classes}")
    log.info(f"Training samples: {len(train_dataset)}")

    aaco_method = create_aaco_method(
        k_neighbors=cfg.aco.k_neighbors,
        acquisition_cost=cfg.aco.acquisition_cost,
        hide_val=cfg.aco.hide_val,
        dataset_name=dataset_type.lower(),
        split=split,
        device=device,
    )

    log.info("Fitting AACO oracle on training data...")
    X_train = train_dataset.features.to(device)
    y_train = train_dataset.labels.to(device)
    aaco_method.aaco_oracle.fit(X_train, y_train)

    log.info("AACO method fitted and ready for evaluation")

    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        aaco_method.save(temp_path)

        split_idx = cfg.dataset_artifact_name.split("_split_")[-1].split(":")[0]

        trained_method_artifact = wandb.Artifact(
            name=f"aaco-{dataset_type}_split_{split_idx}",
            type="trained_method",
            metadata={
                "method_type": "aaco",
                "dataset_type": dataset_type,
                "dataset_artifact_name": cfg.dataset_artifact_name,
                "budget": None,  # AACO doesn't have fixed training budget
                "seed": cfg.seed,
            },
        )
        trained_method_artifact.add_dir(str(temp_path))
        run.log_artifact(trained_method_artifact, aliases=cfg.output_artifact_aliases)

    log.info("AACO method saved as artifact")
    run.finish()


if __name__ == "__main__":
    main()
