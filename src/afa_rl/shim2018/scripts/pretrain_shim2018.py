import argparse
import hydra
from jaxtyping import Float
import os
from types import SimpleNamespace

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.tuner.tuning import Tuner
from matplotlib.figure import Figure
import torch
from torch import Tensor
import yaml
from lightning.pytorch.loggers import WandbLogger

import wandb
from afa_rl.shim2018.models import (
    CopiedSetEncoder,
    CopiedShim2018Embedder,
    Shim2018MLPClassifier,
    ReadProcessEncoder,
    Shim2018Embedder,
    LitShim2018EmbedderClassifier,
    Shim2018MaskedClassifier,
)
from afa_rl.utils import check_masked_classifier_performance, resolve_dataset_config
from common.config_classes import Shim2018PretrainConfig
from common.custom_types import AFADataset
from afa_rl.datasets import DataModuleFromDatasets
from common.utils import dict_to_namespace, get_class_probabilities, set_seed
from pathlib import Path


def get_shim2018_model_from_config(
    cfg: Shim2018PretrainConfig,
    n_features: int,
    n_classes: int,
    class_probabiities: Float[Tensor, "n_classes"],
) -> LitShim2018EmbedderClassifier:
    encoder = ReadProcessEncoder(
        set_element_size=n_features + 1,  # state contains one value and one index
        output_size=cfg.encoder.output_size,
        reading_block_cells=cfg.encoder.reading_block_cells,
        writing_block_cells=cfg.encoder.writing_block_cells,
        memory_size=cfg.encoder.memory_size,
        processing_steps=cfg.encoder.processing_steps,
        dropout=cfg.encoder.dropout,
    )
    embedder = Shim2018Embedder(encoder)
    classifier = Shim2018MLPClassifier(
        cfg.encoder.output_size, n_classes, cfg.classifier.num_cells
    )
    lit_model = LitShim2018EmbedderClassifier(
        embedder=embedder,
        classifier=classifier,
        class_probabilities=class_probabiities,
        max_masking_probability=cfg.max_masking_probability,
        lr=cfg.lr,
    )
    return lit_model


@hydra.main(version_base=None)
def main(cfg: Shim2018PretrainConfig) -> None:
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")

    # Load dataset artifact
    dataset_artifact = wandb.use_artifact(cfg.dataset_artifact_name, type="dataset")
    dataset_artifact_dir = Path(dataset_artifact.download())
    # The dataset dir should contain the files train.pt, val.pt and test.pt
    assert {"train.pt", "val.pt", "test.pt"}.issubset(dataset_artifact_dir.iterdir()), (
        "Dataset artifact must contain train.pt, val.pt and test.pt files."
    )

    # Import is delayed until now to avoid circular imports
    from common.registry import AFA_DATASET_REGISTRY

    train_dataset: AFADataset = AFA_DATASET_REGISTRY[
        dataset_artifact.metadata.class_name
    ].load(dataset_artifact_dir / "train.pt")
    val_dataset: AFADataset = AFA_DATASET_REGISTRY[
        dataset_artifact.metadata.class_name
    ].load(dataset_artifact_dir / "val.pt")
    datamodule = DataModuleFromDatasets(
        train_dataset, val_dataset, batch_size=cfg.batch_size
    )

    # Get the number of features and classes from the dataset
    n_features = train_dataset.features.shape[-1]
    n_classes = train_dataset.labels.shape[-1]

    train_class_probabilities = get_class_probabilities(train_dataset.labels)
    print(f"Class probabilities in training set: {train_class_probabilities}")
    lit_model = get_shim2018_model_from_config(
        cfg, n_features, n_classes, train_class_probabilities
    )
    lit_model = lit_model.to(cfg.device)

    # class_weights = 1 / train_class_probabilities
    # class_weights = class_weights / class_weights.sum()
    # Check accuracy before starting training
    # check_masked_classifier_performance(
    #     masked_classifier=Shim2018MaskedClassifier(lit_model),
    #     dataset=val_dataset,
    #     class_weights=class_weights,
    # )

    # ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss_full",  # Replace "val_loss" with the appropriate validation metric
        save_top_k=1,
        mode="min",
        # dirpath=pretrained_model_path,
        # filename="best-checkpoint",
    )

    run = wandb.init(config=cfg)

    logger = WandbLogger()
    # logger = WandbLogger(project=config.wandb.project, save_dir="logs/afa_rl")
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        logger=logger,
        accelerator=cfg.device,
        devices=1,  # Use only 1 GPU
        callbacks=[checkpoint_callback],
    )

    try:
        trainer.fit(lit_model, datamodule)
    except KeyboardInterrupt:
        pass
    finally:
        # Save best model as wandb artifact
        best_checkpoint = trainer.checkpoint_callback.best_model_path
        pretrained_model_artifact = wandb.Artifact(
            name=f"shim2018-{cfg.dataset_artifact_name}", type="pretrained_model"
        )
        pretrained_model_artifact.add_file(local_path=best_checkpoint)
        run.log_artifact(pretrained_model_artifact)
        run.finish()


if __name__ == "__main__":
    main()
