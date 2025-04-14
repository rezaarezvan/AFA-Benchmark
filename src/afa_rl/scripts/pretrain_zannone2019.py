import argparse
import os
from functools import partial

import lightning as pl
import torch
import yaml
from lightning.pytorch.loggers import WandbLogger
from torch import nn
from torchrl.modules import MLP
from torchvision import transforms

import wandb
from afa_rl.datasets import (
    DataModuleFromDatasets,
    MNISTDataModule,
    Zannone2019CubeDataset,
)
from afa_rl.models import (
    PartialVAE,
    PointNet,
    PointNetType,
    Zannone2019PretrainingModel,
)
from afa_rl.utils import dict_to_namespace, get_1D_identity, get_2D_identity
from common.custom_types import AFADataset
from common.registry import AFA_DATASET_REGISTRY


def get_zannone2019_model_from_config(config):
    # Two different datasets possible: "cube" or "mnist"
    # This will also influence which PointNet to use
    if config.dataset.name == "cube":
        image_shape = None
        assert config.pointnet.naive_identity_type == "1D"
        naive_identity_fn = get_1D_identity
        naive_identity_size = config.dataset.n_features  # onehot
    elif config.dataset.name == "mnist":
        image_shape = (28, 28)
        # For 2D images we can either treat them as 2D coordinates or as 1D features
        if config.pointnet.naive_identity_type == "1D":
            naive_identity_fn = get_1D_identity
            naive_identity_size = config.dataset.n_features
        elif config.pointnet.naive_identity_type == "2D":
            naive_identity_fn = partial(get_2D_identity, image_shape=image_shape)
            naive_identity_size = 2  # coordinates
        else:
            raise ValueError(
                f"Naive identity type {config.pointnet.naive_identity_type} not supported. Use '1D' or '2D'."
            )
    else:
        raise ValueError(
            f"Dataset {config.dataset.name} not supported. Use 'cube' or 'mnist'."
        )

    # PointNet or PointNetPlus
    if config.pointnet.type == "pointnet":
        pointnet_type = PointNetType.POINTNET
        feature_map_encoder_input_size = config.pointnet.identity_size + 1
    elif config.pointnet.type == "pointnetplus":
        pointnet_type = PointNetType.POINTNETPLUS
        feature_map_encoder_input_size = config.pointnet.identity_size
    else:
        raise ValueError(
            f"PointNet type {config.pointnet.type} not supported. Use 'pointnet' or 'pointnetplus'."
        )

    pointnet = PointNet(
        naive_identity_fn=naive_identity_fn,
        identity_network=MLP(
            in_features=naive_identity_size,
            out_features=config.pointnet.identity_size,
            num_cells=config.pointnet.identity_network_num_cells,
            activation_class=nn.ReLU,
        ),
        feature_map_encoder=MLP(
            in_features=feature_map_encoder_input_size,
            out_features=config.pointnet.output_size,
            num_cells=config.pointnet.feature_map_encoder_num_cells,
            activation_class=nn.ReLU,
        ),
        pointnet_type=pointnet_type,
    )
    encoder = MLP(
        in_features=config.pointnet.output_size,
        out_features=2 * config.partial_vae.latent_size,
        num_cells=config.encoder.num_cells,
        activation_class=nn.ReLU,
    )
    partial_vae = PartialVAE(
        pointnet=pointnet,
        encoder=encoder,
        decoder=nn.Sequential(
            MLP(
                in_features=config.partial_vae.latent_size,
                out_features=config.dataset.n_features,
                num_cells=config.partial_vae.decoder_num_cells,
                activation_class=nn.ReLU,
            ),
            nn.Sigmoid() if config.dataset.bounded_0_1 else nn.Identity(),
        ),
    )
    model = Zannone2019PretrainingModel(
        partial_vae=partial_vae,
        # Classifier acts on latent space
        classifier=MLP(
            in_features=config.partial_vae.latent_size,
            out_features=config.classifier.output_size,
            num_cells=config.classifier.num_cells,
            activation_class=nn.ReLU,
        ),
        lr=config.lr,
        verbose=config.verbose,
        image_shape=image_shape,
        max_masking_probability=config.max_masking_probability,
        recon_loss_type=config.recon_loss_type,
        kl_scaling_factor=config.kl_scaling_factor,
        validation_masking_probability=config.validation_masking_probability,
    )
    return model


def main():
    torch.set_float32_matmul_precision("medium")

    # Use argparse to choose config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
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

    train_dataset: AFADataset = AFA_DATASET_REGISTRY[config.dataset.name].load(
        config.dataset.train_path
    )
    val_dataset: AFADataset = AFA_DATASET_REGISTRY[config.dataset.name].load(
        config.dataset.val_path
    )
    datamodule = DataModuleFromDatasets(
        train_dataset, val_dataset, batch_size=config.dataloader.batch_size
    )

    model = get_zannone2019_model_from_config(config)

    logger = WandbLogger(project=config.wandb.project, save_dir="logs/afa_rl")
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        logger=logger,
        accelerator=config.device,
        devices=1,  # Use only 1 GPU
    )
    try:
        trainer.fit(model, datamodule)
    except KeyboardInterrupt:
        pass
    finally:
        # Move the best checkpoint to the desired location
        best_checkpoint = trainer.checkpoint_callback.best_model_path
        # Create parent directories if neccessary
        os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
        # Save
        torch.save(torch.load(best_checkpoint), config.checkpoint_path)
        print(f"Zannone2019PretrainingModel saved to {config.checkpoint_path}")


if __name__ == "__main__":
    main()
