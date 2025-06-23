from jaxtyping import Float
from torch import Tensor, nn
import torch
from torchrl.modules import MLP

from afa_rl.utils import (
    get_1D_identity,
)
from afa_rl.zannone2019.models import (
    PartialVAE,
    PartialVAELossType,
    PointNet,
    PointNetType,
    Zannone2019PretrainingModel,
)
from common.config_classes import Zannone2019PretrainConfig


def get_zannone2019_model_from_config(
    cfg: Zannone2019PretrainConfig,
    n_features: int,
    n_classes: int,
    class_probabilities: Float[Tensor, "n_classes"],
):
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
        identity_size=cfg.pointnet.identity_size,
        n_features=n_features,
        max_embedding_norm=cfg.pointnet.max_embedding_norm,
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
        classifier_loss_scaling_factor=cfg.classifier_loss_scaling_factor,
    )
    return model
