import gc
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

from afabench.afa_rl.datasets import DataModuleFromDatasets
from afabench.afa_rl.shim2018.utils import get_shim2018_model_from_config
from afabench.common.config_classes import Shim2018PretrainConfig
from afabench.common.utils import (
    get_class_frequencies,
    initialize_wandb_run,
    load_dataset_artifact,
    set_seed,
)

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/pretrain/shim2018",
    config_name="config",
)
def main(cfg: Shim2018PretrainConfig) -> None:
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    if cfg.use_wandb:
        run = initialize_wandb_run(
            cfg=cfg, job_type="pretraining", tags=["shim2018"]
        )
    else:
        run = None

    log.info("Loading datasets...")
    train_dataset, train_dataset_metadata = load_dataset_artifact(
        Path(cfg.dataset_artifact_path),
        split="train",
    )
    train_features, train_labels = train_dataset.get_all_data()
    val_dataset, val_dataset_metadata = load_dataset_artifact(
        Path(cfg.dataset_artifact_path),
        split="val",
    )
    datamodule = DataModuleFromDatasets(
        train_dataset, val_dataset, batch_size=cfg.batch_size
    )
    log.info("Loaded datasets.")

    log.info("Creating model...")
    train_class_probabilities = get_class_frequencies(train_labels)
    assert len(train_dataset.feature_shape) == 1, "Only 1D features supported"
    assert len(train_dataset.label_shape) == 1, "Only 1D label supported"
    lit_model = get_shim2018_model_from_config(
        cfg,
        n_features=train_dataset.feature_shape.numel(),
        n_classes=train_dataset.label_shape.numel(),
        class_probabilities=train_class_probabilities,
    )
    lit_model = lit_model.to(cfg.device)
    log.info("Created model.")

    log.info("Starting training...")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss_many_observations",
        save_top_k=1,
        mode="min",
    )
    logger = WandbLogger(save_dir="extra/wandb") if cfg.use_wandb else False
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        logger=logger,
        accelerator=cfg.device,
        devices=1,
        callbacks=[checkpoint_callback],
        limit_train_batches=cfg.limit_train_batches,
        limit_val_batches=cfg.limit_val_batches,
    )

    try:
        trainer.fit(lit_model, datamodule)
    except KeyboardInterrupt:
        pass
    finally:
        log.info("Finished training.")

        log.info("Saving best model...")
        best_checkpoint_path = Path(
            trainer.checkpoint_callback.best_model_path  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
        )

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            ckpt = torch.load(best_checkpoint_path, map_location="cpu")
            torch.save(ckpt, tmp_path / "model.pt")

            metadata = {
                "model_class_name": "Shim2018EmbedderClassifier",
                "dataset_class_name": train_dataset_metadata["class_name"],
                "dataset_artifact_path": cfg.dataset_artifact_path,
                "seed": cfg.seed,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
            with (Path(cfg.save_path) / "metadata.json").open("w") as f:
                json.dump(metadata, f, indent=4)
        log.info("Saved best model.")

        if run is not None:
            run.finish()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


if __name__ == "__main__":
    main()
