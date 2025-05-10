import argparse
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
from common.custom_types import AFADataset
from afa_rl.datasets import DataModuleFromDatasets
from common.models import LitMaskedClassifier, MaskedMLPClassifier
from common.utils import dict_to_namespace, get_class_probabilities, set_seed
from pathlib import Path


def main(config_path: Path, dataset_type: str, train_dataset_path: Path, val_dataset_path: Path, save_path: Path, seed: int):
    set_seed(seed)
    torch.set_float32_matmul_precision("medium")

    # Load config from yaml file
    with open(config_path, "r") as file:
        config_dict: dict = yaml.safe_load(file)

    config = dict_to_namespace(config_dict)

    # Import is delayed until now to avoid circular imports
    from common.registry import AFA_DATASET_REGISTRY
    train_dataset: AFADataset = AFA_DATASET_REGISTRY[dataset_type].load(
        train_dataset_path
    )
    val_dataset: AFADataset = AFA_DATASET_REGISTRY[dataset_type].load(
        val_dataset_path
    )
    datamodule = DataModuleFromDatasets(
        train_dataset, val_dataset, batch_size=config.dataloader.batch_size
    )

    # Get the number of features and classes from the dataset
    n_features = train_dataset.features.shape[-1]
    n_classes = train_dataset.labels.shape[-1]

    model = MaskedMLPClassifier(
        n_features=n_features,
        n_classes=n_classes,
    )
    lit_model = LitMaskedClassifier(
        classifier=model,
        class_probabilities=get_class_probabilities(
            train_dataset.labels
        ),
        max_masking_probability=config.max_masking_probability,
        lr=config.lr,
    )


    # ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss_optimal",
        save_top_k=1,
        mode="min",
        dirpath=save_path,
        filename="best-checkpoint"
    )

    # LearningRateMonitor callback
    # lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # ReduceLROnPlateau scheduler
    # lr_scheduler = {
    #     'scheduler': ReduceLROnPlateau(optimizer=lit_model.optimizer, mode='min', factor=0.1, patience=50),
    #     'monitor': 'val_loss_half',
    # }

    logger = WandbLogger(project=config.wandb.project, save_dir="logs/afa_rl")
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        logger=logger,
        accelerator=config.device,
        devices=1,  # Use only 1 GPU
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
    )

    run = wandb.init(
        entity=config_dict["wandb"]["entity"],
        project=config_dict["wandb"]["project"],
        config=config_dict,
    )
    try:
        trainer.fit(lit_model, datamodule, ckpt_path=None)
    except KeyboardInterrupt:
        pass
    finally:
        run.finish()
        # Move the best checkpoint to the desired location
        best_checkpoint = trainer.checkpoint_callback.best_model_path
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(torch.load(best_checkpoint), save_path / "model.pt")
        # Save params.yml file
        with open(save_path / "params.yml", "w") as file:
            yaml.dump({
                "dataset_type": dataset_type,
                "train_dataset_path": str(train_dataset_path),
                "val_dataset_path": str(val_dataset_path),
                "seed": seed,
            }, file)
        print(f"LitMaskedClassifier saved to {save_path}")


if __name__ == "__main__":
    from common.registry import AFA_DATASET_REGISTRY

    # Use argparse to choose config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=Path, default="configs/common/train_masked_mlp_classifier.yml")
    parser.add_argument("--dataset_type", type=str, default="AFAContext", choices=AFA_DATASET_REGISTRY.keys())
    parser.add_argument("--train_dataset_path", type=Path, default="data/AFAContext/train_split_1.pt")
    parser.add_argument("--val_dataset_path", type=Path, default="data/AFAContext/val_split_1.pt")
    parser.add_argument("--save_path", type=Path, default="models/classifiers/masked_mlp_classifier_temp", help="Path to folder to save the classifier")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(
        config_path=args.config_path,
        dataset_type=args.dataset_type,
        train_dataset_path=args.train_dataset_path,
        val_dataset_path=args.val_dataset_path,
        save_path=args.save_path,
        seed=args.seed
    )
