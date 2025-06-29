import yaml
import argparse
from datetime import datetime
import logging
import gc
from pathlib import Path
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import wandb
from torchrl.modules import MLP
from afa_generative.afa_methods import EDDI, UniformSampler, IterativeSelector, Ma2018AFAMethod
from afa_generative.utils import MaskLayer
from afa_generative.models import PartialVAE, fc_Net
from afa_generative.datasets import prepare_datasets
from common.config_classes import Ma2018PretrainConfig
from common.utils import set_seed, dict_to_namespace, load_dataset_artifact
from afa_rl.zannone2019.models import PointNet, PointNetType
from afa_rl.utils import get_1D_identity


log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../configs/ma2018",
    config_name="pretrain_ma2018",
)
def main(cfg: Ma2018PretrainConfig):
    log.debug(cfg)
    print(OmegaConf.to_yaml(cfg))

    run = wandb.init(
        group="pretrain_ma2018",
        job_type="pretraining",
        config=OmegaConf.to_container(cfg, resolve=True),  # pyright: ignore
    )
    wandb.define_metric("pvae/train_loss", step_metric="pvae/epoch")
    wandb.define_metric("pvae/val_loss",   step_metric="pvae/epoch")
    wandb.define_metric("predictor/train_loss", step_metric="predictor/epoch")
    wandb.define_metric("predictor/val_loss",   step_metric="predictor/epoch")

    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    
    train_dataset, val_dataset, _, _ = load_dataset_artifact(cfg.dataset_artifact_name)

    train_loader, val_loader, d_in, d_out \
        = prepare_datasets(train_dataset, val_dataset, cfg.batch_size)
    
    naive_identity_fn=get_1D_identity
    naive_identity_size = d_in

    # Train PVAE.
    pointnet = PointNet(
        naive_identity_fn=naive_identity_fn,
        identity_network=MLP(
            in_features=naive_identity_size,
            out_features=cfg.pointnet.identity_size,
            num_cells=cfg.pointnet.identity_network_num_cells,
            activation_class=nn.ReLU,
        ),
        feature_map_encoder=MLP(
            in_features=cfg.pointnet.identity_size,
            out_features=cfg.pointnet.output_size,
            num_cells=cfg.pointnet.feature_map_encoder_num_cells,
            activation_class=nn.ReLU,
        ),
        pointnet_type=PointNetType.POINTNETPLUS,
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
    mask_layer = MaskLayer(append=True)
    model = fc_Net(
        input_dim=d_in * 2,
        output_dim=d_out,
        hidden_layer_num=2,
        hidden_unit=[128, 128],
        activations='ReLU',
        drop_out_rate=0.3,
        flag_drop_out=True,
        flag_only_output_layer=False
    )
    sampler = UniformSampler(train_dataset.features)
    iterative = IterativeSelector(model, mask_layer, sampler).to(device)
    iterative.fit(
        train_loader,
        val_loader,
        lr=cfg.classifier.lr,
        nepochs=cfg.classifier.epochs,
        loss_fn=nn.CrossEntropyLoss(),
        patience=cfg.classifier.patience,
        verbose=True)

    eddi_selector = Ma2018AFAMethod(pv, model)
    pretrained_model_artifact = wandb.Artifact(
        name=f"pretrain_ma2018-{cfg.dataset_artifact_name.split(':')[0]}",
        type="pretrained_model",
    )
    eddi_selector.save(Path(cfg.pretrained_model_path))

    model_file = Path(cfg.pretrained_model_path) / "model.pt"
    pretrained_model_artifact.add_file(str(model_file), name="model.pt")
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
