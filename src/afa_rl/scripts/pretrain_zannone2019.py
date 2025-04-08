import argparse
import os

import lightning as pl
import torch
from torchrl.modules import MLP
import yaml
from lightning.pytorch.loggers import WandbLogger

import wandb
from afa_rl.datasets import DataModuleFromDataset
from afa_rl.models import (
    PartialVAE,
    PointNet1D,
    Zannone2019PretrainingModel,
)
from afa_rl.utils import dict_to_namespace
from common.datasets import CubeDataset


def main():
    torch.set_float32_matmul_precision("medium")

    # Use argparse to choose config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    # Load config from yaml file
    with open(args.config, "r") as file:
        config_dict: dict = yaml.safe_load(file)

    wandb.init(
        entity=config_dict["wandb"]["entity"],
        project=config_dict["wandb"]["project"],
        config=config_dict,
    )

    config = dict_to_namespace(config_dict)

    dataset = CubeDataset(
        n_features=config.n_features,
        data_points=config.dataset.size,
        sigma=config.dataset.sigma,
        seed=config.dataset.seed,
    )
    dataset.generate_data()
    datamodule = DataModuleFromDataset(
        dataset=dataset,
        batch_size=config.dataloader.batch_size,
        train_ratio=config.dataloader.train_ratio,
        num_workers=config.dataloader.num_workers,
    )

    pointnet = PointNet1D(
        element_encoder=MLP(
            in_features=config.n_features + 1,
            out_features=config.pointnet.output_size,
            num_cells=config.pointnet.num_cells,
        ),
    )
    encoder = MLP(
        in_features=config.pointnet.output_size,
        out_features=config.encoder.output_size,
        num_cells=config.encoder.num_cells,
    )
    partial_vae = PartialVAE(
        pointnet=pointnet,
        encoder=encoder,
        mu_net=MLP(
            in_features=config.encoder.output_size,
            out_features=config.partial_vae.latent_size,
            num_cells=config.partial_vae.mu_net_num_cells,
        ),
        logvar_net=MLP(
            in_features=config.encoder.output_size,
            out_features=config.partial_vae.latent_size,
            num_cells=config.partial_vae.logvar_net_num_cells,
        ),
        decoder=MLP(
            in_features=config.partial_vae.latent_size,
            out_features=config.n_features,
            num_cells=config.partial_vae.decoder_num_cells,
        ),
    )
    model = Zannone2019PretrainingModel(
        partial_vae=partial_vae,
        classifier=MLP(
            in_features=config.encoder.output_size,
            out_features=8,
            num_cells=config.classifier.num_cells,
        ),
        lr=config.lr,
    )

    logger = WandbLogger(project=config.wandb.project, save_dir="logs/afa_rl")
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        logger=logger,
        accelerator=config.device,
        devices=1,  # Use only 1 GPU
        strategy="ddp",
    )
    try:
        trainer.fit(model, datamodule)
    except KeyboardInterrupt:
        pass
    finally:
        # Move the best checkpoint to the desired location
        best_checkpoint = trainer.checkpoint_callback.best_model_path
        # Create checkpoints directory if it doesn't exist
        checkpoints_dir = "checkpoints/afa_rl"
        os.makedirs(checkpoints_dir, exist_ok=True)
        # Save
        torch.save(
            torch.load(best_checkpoint), f"checkpoints/afa_rl/{config.checkpoint}"
        )
        print(
            f"Zannone2019PretrainingModel saved to checkpoints/afa_rl/{config.checkpoint}"
        )


if __name__ == "__main__":
    main()
