import argparse
import os
from functools import partial
from pathlib import Path

import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report
import torch
from torch import Tensor
import yaml
from lightning.pytorch.loggers import WandbLogger
from torch import nn
from torchrl.modules import MLP

import wandb
from afa_rl.datasets import (
    DataModuleFromDatasets,
)
from afa_rl.models import (
    PartialVAE,
    PointNet,
    PointNetType,
    Zannone2019PretrainingModel,
)
from afa_rl.utils import dict_to_namespace, get_1D_identity
from common.custom_types import AFADataset
from common.registry import AFA_DATASET_REGISTRY
from jaxtyping import Float

from common.utils import get_class_probabilities, set_seed


def get_zannone2019_model_from_config(config, n_features: int, n_classes: int, class_probabilities: Float[Tensor, "n_classes"]):
    naive_identity_fn = get_1D_identity
    naive_identity_size = n_features  # onehot

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
                out_features=n_features,
                num_cells=config.partial_vae.decoder_num_cells,
                activation_class=nn.ReLU,
            ),
            nn.Identity(),
        ),
    )
    model = Zannone2019PretrainingModel(
        partial_vae=partial_vae,
        # Classifier acts on latent space
        classifier=MLP(
            in_features=config.partial_vae.latent_size,
            out_features=n_classes,
            num_cells=config.classifier.num_cells,
            activation_class=nn.ReLU,
        ),
        lr=config.lr,
        verbose=config.verbose,
        max_masking_probability=config.max_masking_probability,
        class_probabilities=class_probabilities,
        recon_loss_type=config.recon_loss_type,
        kl_scaling_factor=config.kl_scaling_factor,
        validation_masking_probability=config.validation_masking_probability,
    )
    return model


def main(pretrain_config_path: Path, dataset_type: str, train_dataset_path: Path, val_dataset_path: Path, pretrained_model_path: Path, seed: int):
    set_seed(seed)
    torch.set_float32_matmul_precision("medium")

    # Load config from yaml file
    with open(pretrain_config_path, "r") as file:
        config_dict: dict = yaml.safe_load(file)

    config = dict_to_namespace(config_dict)

    train_dataset: AFADataset = AFA_DATASET_REGISTRY[dataset_type].load(
        train_dataset_path
    )
    assert train_dataset.features is not None
    assert train_dataset.labels is not None
    val_dataset: AFADataset = AFA_DATASET_REGISTRY[dataset_type].load(
        val_dataset_path
    )
    assert val_dataset.features is not None
    assert val_dataset.labels is not None
    datamodule = DataModuleFromDatasets(
        train_dataset, val_dataset, batch_size=config.dataloader.batch_size
    )

    n_features = train_dataset.features.shape[-1]
    n_classes = train_dataset.labels.shape[-1]
    train_class_probabilities = get_class_probabilities(train_dataset.labels)
    model = get_zannone2019_model_from_config(config, n_features, n_classes, train_class_probabilities)

    # ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  # Replace "val_loss" with the appropriate validation metric
        save_top_k=1,
        mode="min",
        dirpath=pretrained_model_path,
        filename="best-checkpoint"
    )

    logger = WandbLogger(project=config.wandb.project, save_dir="logs/afa_rl")
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        logger=logger,
        accelerator=config.device,
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
        # Move the best checkpoint to the desired location
        best_checkpoint = trainer.checkpoint_callback.best_model_path
        # Create parent directories if necessary
        os.makedirs(os.path.dirname(pretrained_model_path), exist_ok=True)
        # Save weights
        torch.save(torch.load(best_checkpoint), pretrained_model_path / "model.pt")
        # Save params.yml file
        with open(pretrained_model_path / "params.yml", "w") as file:
            yaml.dump({
                "dataset_type": dataset_type,
                "train_dataset_path": str(train_dataset_path),
                "val_dataset_path": str(val_dataset_path),
                "seed": seed,
            }, file)
        print(f"Zannone2019PretrainingModel saved to {pretrained_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_config", type=str, required=True)
    parser.add_argument("--dataset_type", type=str, required=True, choices=AFA_DATASET_REGISTRY.keys())
    parser.add_argument("--train_dataset_path", type=Path, required=True)
    parser.add_argument("--val_dataset_path", type=Path, required=True)
    parser.add_argument("--pretrained_model_path", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    main(
        pretrain_config_path=args.pretrain_config,
        dataset_type=args.dataset_type,
        train_dataset_path=args.train_dataset_path,
        val_dataset_path=args.val_dataset_path,
        pretrained_model_path=args.pretrained_model_path,
        seed=args.seed
    )
