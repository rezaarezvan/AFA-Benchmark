from datetime import datetime
import gc
import logging
from omegaconf import OmegaConf

import hydra
import lightning as pl
import torch
from jaxtyping import Float
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor, nn
from torchrl.modules import MLP

import wandb
from afa_rl.datasets import (
    DataModuleFromDatasets,
)
from afa_rl.utils import get_1D_identity
from afa_rl.zannone2019.models import (
    PartialVAE,
    PartialVAELossType,
    PointNet,
    PointNetType,
    Zannone2019PretrainingModel,
)
from common.config_classes import Zannone2019PretrainConfig
from common.utils import (
    get_class_probabilities,
    load_dataset_artifact,
    set_seed,
)


def get_zannone2019_model_from_config(
    cfg: Zannone2019PretrainConfig,
    n_features: int,
    n_classes: int,
    class_probabilities: Float[Tensor, "n_classes"],
):
    naive_identity_fn = get_1D_identity
    naive_identity_size = n_features  # onehot

    # PointNet or PointNetPlus
    if cfg.pointnet.type == "pointnet":
        pointnet_type = PointNetType.POINTNET
        feature_map_encoder_input_size = cfg.pointnet.identity_size + 1
    elif cfg.pointnet.type == "pointnetplus":
        pointnet_type = PointNetType.POINTNETPLUS
        feature_map_encoder_input_size = cfg.pointnet.identity_size
    else:
        raise ValueError(
            f"PointNet type {cfg.pointnet.type} not supported. Use 'pointnet' or 'pointnetplus'."
        )

    pointnet = PointNet(
        naive_identity_fn=naive_identity_fn,
        identity_network=MLP(
            in_features=naive_identity_size,
            out_features=cfg.pointnet.identity_size,
            num_cells=cfg.pointnet.identity_network_num_cells,
            dropout=cfg.pointnet.identity_network_dropout,
            activation_class=nn.ReLU,
        ),
        feature_map_encoder=MLP(
            in_features=feature_map_encoder_input_size,
            out_features=cfg.pointnet.output_size,
            num_cells=cfg.pointnet.feature_map_encoder_num_cells,
            dropout=cfg.pointnet.feature_map_encoder_dropout,
            activation_class=nn.ReLU,
        ),
        pointnet_type=pointnet_type,
    )
    encoder = MLP(
        in_features=cfg.pointnet.output_size,
        out_features=2 * cfg.partial_vae.latent_size,
        num_cells=cfg.encoder.num_cells,
        dropout=cfg.encoder.dropout,
        activation_class=nn.ReLU,
    )
    partial_vae = PartialVAE(
        pointnet=pointnet,
        encoder=encoder,
        decoder=nn.Sequential(
            MLP(
                in_features=cfg.partial_vae.latent_size,
                out_features=n_features,
                num_cells=cfg.partial_vae.decoder_num_cells,
                dropout=cfg.partial_vae.decoder_dropout,
                activation_class=nn.ReLU,
            ),
            nn.Identity(),
        ),
    )
    if cfg.recon_loss_type == "squared_error":
        recon_loss_type = PartialVAELossType.SQUARED_ERROR
    elif cfg.recon_loss_type == "binary_cross_entropy":
        recon_loss_type = PartialVAELossType.BINARY_CROSS_ENTROPY
    else:
        raise ValueError(
            f"{cfg.recon_loss_type} was not one of ('squared_error', 'binary_cross_entropy')"
        )
    model = Zannone2019PretrainingModel(
        partial_vae=partial_vae,
        # Classifier acts on latent space
        classifier=MLP(
            in_features=cfg.partial_vae.latent_size,
            out_features=n_classes,
            num_cells=cfg.classifier.num_cells,
            dropout=cfg.classifier.dropout,
            activation_class=nn.ReLU,
        ),
        lr=cfg.lr,
        min_masking_probability=cfg.min_masking_probability,
        max_masking_probability=cfg.max_masking_probability,
        class_probabilities=class_probabilities,
        recon_loss_type=recon_loss_type,
        kl_scaling_factor=cfg.kl_scaling_factor,
    )
    return model


log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../conf/pretrain/zannone2019",
    config_name="config",
)
def main(cfg: Zannone2019PretrainConfig) -> None:
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    run = wandb.init(
        group="pretrain_zannone2019",
        job_type="pretraining",
        config=OmegaConf.to_container(cfg, resolve=True),  # pyright: ignore
    )

    # Load dataset artifact
    train_dataset, val_dataset, _, _ = load_dataset_artifact(cfg.dataset_artifact_name)
    datamodule = DataModuleFromDatasets(
        train_dataset, val_dataset, batch_size=cfg.batch_size
    )

    # Get the number of features and classes from the dataset
    n_features = train_dataset.features.shape[-1]
    n_classes = train_dataset.labels.shape[-1]

    train_class_probabilities = get_class_probabilities(train_dataset.labels)
    log.debug(f"Class probabilities in training set: {train_class_probabilities}")
    lit_model = get_zannone2019_model_from_config(
        cfg, n_features, n_classes, train_class_probabilities
    )
    lit_model = lit_model.to(cfg.device)

    # ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss_many_observations",  # val_loss_few_observations could also work but is probably not as robust
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
        limit_train_batches=cfg.limit_train_batches,
        limit_val_batches=cfg.limit_val_batches,
    )

    try:
        trainer.fit(lit_model, datamodule)
    except KeyboardInterrupt:
        pass
    finally:
        # Save best model as wandb artifact
        best_checkpoint = trainer.checkpoint_callback.best_model_path  # pyright: ignore
        pretrained_model_artifact = wandb.Artifact(
            name=f"pretrain_zannone2019-{cfg.dataset_artifact_name.split(':')[0]}",
            type="pretrained_model",
        )
        pretrained_model_artifact.add_file(local_path=best_checkpoint, name="model.pt")
        run.log_artifact(
            pretrained_model_artifact,
            aliases=[*cfg.output_artifact_aliases, datetime.now().strftime("%b%d")],
        )
        run.finish()

        gc.collect()  # Force Python GC
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Release cached memory held by PyTorch CUDA allocator
            torch.cuda.synchronize()  # Optional, wait for CUDA ops to finish


if __name__ == "__main__":
    main()
