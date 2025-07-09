import yaml
import argparse
from datetime import datetime
import logging
import gc
from pathlib import Path
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tempfile import TemporaryDirectory
import torch
import torch.nn as nn
import wandb
from torchrl.modules import MLP
from afa_generative.afa_methods import UniformSampler, IterativeSelector, EDDI_Training, Ma2018AFAMethod
from afa_generative.utils import MaskLayer
from afa_generative.models import PartialVAE, fc_Net
from afa_generative.datasets import prepare_datasets
from common.config_classes import Ma2018PretraingConfig
from common.utils import set_seed, dict_to_namespace, load_dataset_artifact
from afa_rl.zannone2019.models import PointNet, PointNetType
from afa_rl.utils import get_1D_identity


log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../conf/pretrain/ma2018",
    config_name="config",
)
def main(cfg: Ma2018PretraingConfig):
    log.debug(cfg)
    print(OmegaConf.to_yaml(cfg))

    run = wandb.init(
        group="pretrain_ma2018",
        job_type="pretraining",
        config=OmegaConf.to_container(cfg, resolve=True),  # pyright: ignore
    )
    wandb.define_metric("pvae_pretrain/train_loss", step_metric="pvae_pretrain/epoch")
    wandb.define_metric("pvae_pretrain/val_loss",   step_metric="pvae_pretrain/epoch")
    wandb.define_metric("joint_training/train_loss", step_metric="joint_training/epoch")
    wandb.define_metric("joint_training/val_loss",   step_metric="joint_training/epoch")

    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    
    train_dataset, val_dataset, _, _ = load_dataset_artifact(cfg.dataset_artifact_name)

    train_loader, val_loader, d_in, d_out \
        = prepare_datasets(train_dataset, val_dataset, cfg.batch_size)
    
    # naive_identity_fn=get_1D_identity
    # naive_identity_size = d_in

    # Train PVAE.
    pointnet = PointNet(
        identity_size=cfg.pointnet.identity_size,
        n_features=d_in,
        feature_map_encoder=MLP(
            in_features=cfg.pointnet.identity_size,
            out_features=cfg.pointnet.output_size,
            num_cells=cfg.pointnet.feature_map_encoder_num_cells,
            activation_class=nn.ReLU,
        ),
        pointnet_type=PointNetType.POINTNETPLUS,
        max_embedding_norm=cfg.pointnet.max_embedding_norm,
    )
    encoder = MLP(
        in_features=cfg.pointnet.output_size,
        out_features=2 * cfg.partial_vae.latent_size,
        num_cells=cfg.partial_vae.encoder_num_cells,
        activation_class=nn.ReLU,
    )
    pv = PartialVAE(
        pointnet=pointnet,
        encoder=encoder,
        decoder=nn.Sequential(
            MLP(
                in_features=cfg.partial_vae.latent_size,
                out_features=d_in,
                num_cells=cfg.partial_vae.decoder_num_cells,
                activation_class=nn.ReLU,
            ),
            nn.Identity(),
        ),
    )
    pv = pv.to(device)
    pv.fit(train_loader=train_loader, 
           val_loader=val_loader, 
           lr=cfg.partial_vae.lr, 
           nepochs=cfg.partial_vae.epochs,
           p_max=cfg.partial_vae.max_masking_probability,
           patience=cfg.partial_vae.patience,
           kl_scaling_factor=cfg.partial_vae.kl_scaling_factor)
    
    # Train masked predictor.
    model = MLP(
        in_features=cfg.partial_vae.latent_size,
        out_features=d_out,
        num_cells=cfg.classifier.num_cells,
        dropout=cfg.classifier.dropout,
        activation_class=nn.ReLU,
    )
    eddi = EDDI_Training(model, pv)
    eddi.fit(train_loader, val_loader)

    eddi_selector = Ma2018AFAMethod(pv, model)
    pretrained_model_artifact = wandb.Artifact(
        name=f"pretrain_ma2018-{cfg.dataset_artifact_name.split(':')[0]}",
        type="pretrained_model",
    )
    with TemporaryDirectory(delete=False) as tmp_path_str:
        tmp_path = Path(tmp_path_str)
        eddi_selector.save(tmp_path)
        del eddi_selector
        eddi_selector = Ma2018AFAMethod.load(tmp_path / "model.pt", device=device)

    pretrained_model_artifact.add_file(str(tmp_path / 'model.pt'))
    run.log_artifact(
        pretrained_model_artifact,
        aliases=[*cfg.output_artifact_aliases, datetime.now().strftime("%b%d")]
    )
    run.finish()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
