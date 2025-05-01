import argparse
from jaxtyping import Float
import os
from types import SimpleNamespace

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from torch import Tensor
import yaml
from lightning.pytorch.loggers import WandbLogger

import wandb
from afa_rl.models import (
    ShimMLPClassifier,
    ReadProcessEncoder,
    ShimEmbedder,
    ShimEmbedderClassifier,
)
from afa_rl.utils import dict_to_namespace
from common.custom_types import AFADataset
from afa_rl.datasets import DataModuleFromDatasets
from common.registry import AFA_DATASET_REGISTRY
from common.utils import get_class_probabilities, set_seed


def get_shim2018_model_from_config(config: SimpleNamespace, n_features: int, n_classes: int, class_probabiities: Float[Tensor, "n_classes"]):
    encoder = ReadProcessEncoder(
        feature_size=n_features + 1,  # state contains one value and one index
        output_size=config.encoder.output_size,
        reading_block_cells=config.encoder.reading_block_cells,
        writing_block_cells=config.encoder.writing_block_cells,
        memory_size=config.encoder.memory_size,
        processing_steps=config.encoder.processing_steps,
    )
    embedder = ShimEmbedder(encoder)
    classifier = ShimMLPClassifier(
        config.encoder.output_size, n_classes, config.classifier.num_cells
    )
    model = ShimEmbedderClassifier(
        embedder=embedder,
        classifier=classifier,
        class_probabilities=class_probabiities,
        lr=config.embedder_classifier.lr
    )
    return model


def main(args: argparse.Namespace):
    set_seed(args.seed)
    torch.set_float32_matmul_precision("medium")

    # Load config from yaml file
    with open(args.pretrain_config, "r") as file:
        config_dict: dict = yaml.safe_load(file)

    config = dict_to_namespace(config_dict)

    train_dataset: AFADataset = AFA_DATASET_REGISTRY[args.dataset_type].load(
        args.train_dataset_path
    )
    assert train_dataset.features is not None
    assert train_dataset.labels is not None
    val_dataset: AFADataset = AFA_DATASET_REGISTRY[args.dataset_type].load(
        args.val_dataset_path
    )
    assert val_dataset.features is not None
    assert val_dataset.labels is not None
    datamodule = DataModuleFromDatasets(
        train_dataset, val_dataset, batch_size=config.dataloader.batch_size
    )

    # Get the number of features and classes from the dataset
    n_features = train_dataset.features.shape[-1]
    n_classes = train_dataset.labels.shape[-1]


    train_class_probabilities = get_class_probabilities(train_dataset.labels)
    print(f"Class probabilities in training set: {train_class_probabilities}")
    model = get_shim2018_model_from_config(config, n_features, n_classes, train_class_probabilities)
    model = model.to(config.device)


    # ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  # Replace "val_loss" with the appropriate validation metric
        save_top_k=1,
        mode="min",
        dirpath=args.pretrained_model_path,
        filename="best-checkpoint"
    )

    logger = WandbLogger(project=config.wandb.project, save_dir="logs/afa_rl")
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        logger=logger,
        accelerator=config.device,
        # devices=1,  # Use only 1 GPU
        # strategy="ddp",
        # accelerator=config.device,
        devices=1,  # Use only 1 GPU
        callbacks=[checkpoint_callback],
    )

    run = wandb.init(
        entity=config_dict["wandb"]["entity"],
        project=config_dict["wandb"]["project"],
        config=config_dict,
    )
    try:
        trainer.fit(model, datamodule)
    except KeyboardInterrupt:
        pass
    finally:
        run.finish()
        # Move the best checkpoint to the desired location
        best_checkpoint = trainer.checkpoint_callback.best_model_path
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(args.pretrained_model_path), exist_ok=True)
        torch.save(torch.load(best_checkpoint), args.pretrained_model_path / "model.pt")
        # Save params.yml file
        with open(args.pretrained_model_path / "params.yml", "w") as file:
            yaml.dump({
                "dataset_type": args.dataset_type,
                "train_dataset_path": args.train_dataset_path,
                "val_dataset_path": args.val_dataset_path,
                "seed": args.seed,
            }, file)
        print(f"ShimEmbedderClassifier saved to {args.pretrained_model_path}")


if __name__ == "__main__":
    # Use argparse to choose config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_config", type=str, required=True)
    parser.add_argument("--dataset_type", type=str, required=True, choices=AFA_DATASET_REGISTRY.keys())
    parser.add_argument("--train_dataset_path", type=str, required=True)
    parser.add_argument("--val_dataset_path", type=str, required=True)
    parser.add_argument("--pretrained_model_path", type=str, required=True, help="Path to folder to save the pretrained model")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    main(args)
