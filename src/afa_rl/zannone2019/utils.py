from pathlib import Path
from typing import Any
from dacite import from_dict
from jaxtyping import Float
from torch import Tensor, nn
import torch
from torchrl.modules import MLP
import wandb

from afa_rl.utils import (
    get_1D_identity,
    str_to_activation_class_mapping,
)
from afa_rl.zannone2019.models import (
    PartialVAE,
    PartialVAELossType,
    PointNet,
    PointNetType,
    Zannone2019PretrainingModel,
)
from common.config_classes import Zannone2019PretrainConfig
from common.custom_types import AFADataset
from common.utils import get_class_probabilities, load_dataset_artifact


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
        n_features=n_features + n_classes
        if cfg.reconstruct_label
        else n_features,  # since we append the one-hot label to the features
        max_embedding_norm=cfg.pointnet.max_embedding_norm,
        feature_map_encoder=MLP(
            in_features=feature_map_encoder_input_size,
            out_features=cfg.pointnet.output_size,
            num_cells=cfg.pointnet.feature_map_encoder_num_cells,
            dropout=cfg.pointnet.feature_map_encoder_dropout,
            activation_class=str_to_activation_class_mapping[
                cfg.pointnet.feature_map_encoder_activation_class
            ],
        ),
        pointnet_type=pointnet_type,
    )
    encoder = MLP(
        in_features=cfg.pointnet.output_size,
        out_features=2 * cfg.partial_vae.latent_size,
        num_cells=cfg.encoder.num_cells,
        dropout=cfg.encoder.dropout,
        activation_class=str_to_activation_class_mapping[cfg.encoder.activation_class],
    )
    partial_vae = PartialVAE(
        pointnet=pointnet,
        encoder=encoder,
        decoder=MLP(
            in_features=cfg.partial_vae.latent_size,
            out_features=n_features + n_classes
            if cfg.reconstruct_label
            else n_features,  # since we append the one-hot label to the features
            num_cells=cfg.partial_vae.decoder_num_cells,
            dropout=cfg.partial_vae.decoder_dropout,
            activation_class=str_to_activation_class_mapping[
                cfg.partial_vae.decoder_activation_class
            ],
        ),
    )
    model = Zannone2019PretrainingModel(
        partial_vae=partial_vae,
        # Classifier acts on latent space
        classifier=MLP(
            in_features=cfg.partial_vae.latent_size,
            out_features=n_classes,
            num_cells=cfg.classifier.num_cells,
            dropout=cfg.classifier.dropout,
            activation_class=str_to_activation_class_mapping[
                cfg.classifier.activation_class
            ],
        ),
        lr=cfg.lr,
        min_masking_probability=cfg.min_masking_probability,
        max_masking_probability=cfg.max_masking_probability,
        class_probabilities=class_probabilities,
        start_kl_scaling_factor=cfg.start_kl_scaling_factor,
        end_kl_scaling_factor=cfg.end_kl_scaling_factor,
        n_annealing_epochs=cfg.n_annealing_epochs,
        classifier_loss_scaling_factor=cfg.classifier_loss_scaling_factor,
        reconstruct_label=cfg.reconstruct_label,
        label_loss_scaling_factor=cfg.label_loss_scaling_factor,
    )
    return model


def load_pretrained_model_artifacts(
    artifact_name: str,
) -> tuple[
    AFADataset,  # train dataset
    AFADataset,  # val dataset
    AFADataset,  # test dataset
    dict[str, Any],  # dataset metadata
    Zannone2019PretrainingModel,
    Zannone2019PretrainConfig,
]:
    """Load a pretrained model and the dataset it was trained on, from a WandB artifact."""
    pretrained_model_artifact = wandb.use_artifact(
        artifact_name, type="pretrained_model"
    )
    pretrained_model_artifact_dir = Path(pretrained_model_artifact.download())
    # The dataset dir should contain a file called model.pt
    artifact_filenames = [f.name for f in pretrained_model_artifact_dir.iterdir()]
    assert {"model.pt"}.issubset(artifact_filenames), (
        f"Dataset artifact must contain a model.pt file. Instead found: {artifact_filenames}"
    )

    # Access config of the run that produced this pretrained model
    pretraining_run = pretrained_model_artifact.logged_by()
    assert pretraining_run is not None, (
        "Pretrained model artifact must be logged by a run."
    )
    pretrained_model_config_dict = pretraining_run.config
    pretrained_model_config: Zannone2019PretrainConfig = from_dict(
        data_class=Zannone2019PretrainConfig, data=pretrained_model_config_dict
    )

    # Load the dataset that the pretrained model was trained on
    train_dataset, val_dataset, test_dataset, dataset_metadata = load_dataset_artifact(
        pretrained_model_config.dataset_artifact_name
    )

    # Get the number of features and classes from the dataset
    n_features = train_dataset.features.shape[-1]
    n_classes = train_dataset.labels.shape[-1]
    train_class_probabilities = get_class_probabilities(train_dataset.labels)

    pretrained_model = get_zannone2019_model_from_config(
        pretrained_model_config,
        n_features,
        n_classes,
        train_class_probabilities,
    )
    pretrained_model_checkpoint = torch.load(
        pretrained_model_artifact_dir / "model.pt",
        weights_only=True,
        map_location=torch.device("cpu"),
    )
    pretrained_model.load_state_dict(pretrained_model_checkpoint["state_dict"])

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        dataset_metadata,
        pretrained_model,
        pretrained_model_config,
    )
