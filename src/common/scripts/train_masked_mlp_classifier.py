import argparse
import logging
from tempfile import NamedTemporaryFile, TemporaryDirectory, TemporaryFile
from typing import cast
import hydra
from jaxtyping import Float
import os
from types import SimpleNamespace

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import torch
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
from lightning.pytorch.loggers import WandbLogger

import wandb
from afa_rl.shim2018.models import (
    Shim2018MLPClassifier,
    ReadProcessEncoder,
    Shim2018Embedder,
    LitShim2018EmbedderClassifier,
)
from common.classifiers import WrappedMaskedMLPClassifier
from common.config_classes import TrainMaskedMLPClassifierConfig
from common.custom_types import AFADataset
from afa_rl.datasets import DataModuleFromDatasets
from common.models import LitMaskedClassifier, MaskedMLPClassifier
from common.utils import dict_to_namespace, get_class_probabilities, load_dataset_artifact, set_seed
from pathlib import Path
from omegaconf import OmegaConf

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../../../conf/classifiers/masked_mlp_classifier", config_name="tmp")
def main(cfg: TrainMaskedMLPClassifierConfig) -> None:
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")

    run = wandb.init(
        group="train_masked_mlp_classifier",
        job_type="train_classifier",
        config=OmegaConf.to_container(cfg, resolve=True),  # pyright: ignore
    )

    # Load dataset artifact
    train_dataset, val_dataset, _ = load_dataset_artifact(cfg.dataset_artifact.name)
    datamodule = DataModuleFromDatasets(
        train_dataset, val_dataset, batch_size=cfg.batch_size
    )

    # Get the number of features and classes from the dataset
    n_features = train_dataset.features.shape[-1]
    n_classes = train_dataset.labels.shape[-1]

    train_class_probabilities = get_class_probabilities(train_dataset.labels)
    log.debug(f"Class probabilities in training set: {train_class_probabilities}")
    model = MaskedMLPClassifier(
        n_features=n_features,
        n_classes=n_classes,
    )
    lit_model = LitMaskedClassifier(
        classifier=model,
        class_probabilities=train_class_probabilities,
        max_masking_probability=cfg.max_masking_probability,
        lr=cfg.lr,
    )


    # ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss_full",
        save_top_k=1,
        mode="min",
    )

    logger = WandbLogger()
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        logger=logger,
        accelerator=cfg.device,
        devices=1,  # Use only 1 GPU
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
    )

    try:
        trainer.fit(lit_model, datamodule, ckpt_path=None)
    except KeyboardInterrupt:
        pass
    finally:
        # Convert lightning model to a classifier that implements the AFAClassifier interface
        best_checkpoint = trainer.checkpoint_callback.best_model_path
        best_lit_model = LitMaskedClassifier.load_from_checkpoint(
            best_checkpoint,
            classifier=model,
            class_probabilities=train_class_probabilities,
            max_masking_probability=cfg.max_masking_probability,
            lr=cfg.lr,
        )
        best_model = cast(MaskedMLPClassifier, best_lit_model.classifier)
        wrapped_classifier = WrappedMaskedMLPClassifier(model=best_model, device=torch.device(cfg.device))
        # Save model in a temporary file
        with NamedTemporaryFile(delete=False) as tmp_file:
            save_path = Path(tmp_file.name)
            wrapped_classifier.save(save_path)
        # Save classifier as wandb artifact
        trained_classifier_artifact = wandb.Artifact(
            name=f"masked_mlp_classifier-{cfg.dataset_artifact.name.split(':')[0]}",
            type="trained_classifier",
            metadata={
                "dataset_name": cfg.dataset_artifact.name.split(":")[0],
                "seed": cfg.seed,
                "classifier_class": wrapped_classifier.__class__.__name__,
            },
        )
        trained_classifier_artifact.add_file(str(save_path), name="classifier.pt")
        run.log_artifact(trained_classifier_artifact)
        run.finish()


if __name__ == "__main__":
    main()
