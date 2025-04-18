import argparse

import lightning as pl
import torch
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


def get_shim2018_model_from_config(config):
    encoder = ReadProcessEncoder(
        feature_size=config.n_features + 1,  # state contains one value and one index
        output_size=config.encoder.output_size,
        reading_block_cells=config.encoder.reading_block_cells,
        writing_block_cells=config.encoder.writing_block_cells,
        memory_size=config.encoder.memory_size,
        processing_steps=config.encoder.processing_steps,
    )
    embedder = ShimEmbedder(encoder)
    classifier = ShimMLPClassifier(
        config.encoder.output_size, 8, config.classifier.num_cells
    )
    model = ShimEmbedderClassifier(
        embedder=embedder, classifier=classifier, lr=config.embedder_classifier.lr
    )
    return model


def main(args: argparse.Namespace):
    torch.set_float32_matmul_precision("medium")

    # Load config from yaml file
    with open(args.config, "r") as file:
        config_dict: dict = yaml.safe_load(file)

    config = dict_to_namespace(config_dict)

    # dataset = CubeDataset(
    #     n_features=config.n_features,
    #     data_points=config.dataset.size,
    #     sigma=config.dataset.sigma,
    #     seed=config.dataset.seed,
    # )
    # datamodule = DataModuleFromDatasets(
    #     dataset=dataset,
    #     batch_size=config.dataloader.batch_size,
    #     train_ratio=config.dataloader.train_ratio,
    #     num_workers=config.dataloader.num_workers,
    # )
    train_dataset: AFADataset = AFA_DATASET_REGISTRY[config.dataset.name].load(
        config.dataset.train_path
    )
    val_dataset: AFADataset = AFA_DATASET_REGISTRY[config.dataset.name].load(
        config.dataset.val_path
    )
    datamodule = DataModuleFromDatasets(
        train_dataset, val_dataset, batch_size=config.dataloader.batch_size
    )

    model = get_shim2018_model_from_config(config)
    model = model.to(config.device)

    run = wandb.init(
        entity=config_dict["wandb"]["entity"],
        project=config_dict["wandb"]["project"],
        config=config_dict,
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
    )
    try:
        trainer.fit(model, datamodule)
    except KeyboardInterrupt:
        pass
    finally:
        run.finish()
        # Move the best checkpoint to the desired location
        best_checkpoint = trainer.checkpoint_callback.best_model_path
        torch.save(torch.load(best_checkpoint), config.checkpoint_path)
        print(f"ShimEmbedderClassifier saved to {config.checkpoint_path}")


if __name__ == "__main__":
    # Use argparse to choose config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    main(args)
