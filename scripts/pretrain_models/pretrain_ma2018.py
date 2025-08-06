from datetime import datetime
import logging
import gc
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from tempfile import TemporaryDirectory
import torch
import torch.nn as nn
import wandb
from torchrl.modules import MLP
from afa_generative.afa_methods import EDDI_Training, Ma2018AFAMethod
from afa_generative.models import PartialVAE
from afa_generative.datasets import prepare_datasets
from common.config_classes import Ma2018PretrainingConfig
from common.utils import set_seed, load_dataset_artifact
from afa_rl.zannone2019.models import PointNet, PointNetType


log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../conf/pretrain/ma2018",
    config_name="config",
)
def main(cfg: Ma2018PretrainingConfig):
    log.debug(cfg)
    print(OmegaConf.to_yaml(cfg))

    run = wandb.init(
        group="pretrain_ma2018",
        job_type="pretraining",
        config=OmegaConf.to_container(cfg, resolve=True),  # pyright: ignore
        tags=["EDDI"],
        dir="wandb",
    )

    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    train_dataset, val_dataset, _, _ = load_dataset_artifact(cfg.dataset_artifact_name)

    train_loader, val_loader, d_in, d_out = prepare_datasets(
        train_dataset, val_dataset, cfg.batch_size
    )

    # Train PVAE.
    pointnet = PointNet(
        identity_size=cfg.pointnet.identity_size,
        n_features=d_in + d_out,
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
        decoder=MLP(
            in_features=cfg.partial_vae.latent_size,
            out_features=d_in + d_out,
            num_cells=cfg.partial_vae.decoder_num_cells,
            activation_class=nn.ReLU,
        ),
    )
    pv = pv.to(device)

    classifier = MLP(
        in_features=cfg.partial_vae.latent_size,
        out_features=d_out,
        num_cells=cfg.classifier.num_cells,
        dropout=cfg.classifier.dropout,
        activation_class=nn.ReLU,
    )
    eddi_model = EDDI_Training(
        classifier=classifier,
        partial_vae=pv,
        num_classes=d_out,
        n_annealing_epochs=cfg.n_annealing_epochs,
        start_kl_scaling_factor=cfg.start_kl_scaling_factor,
        end_kl_scaling_factor=cfg.end_kl_scaling_factor,
    )
    eddi_model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        lr=cfg.classifier.lr,
        classifier_loss_scaling_factor=cfg.classifier.classifier_loss_scaling_factor,
        min_mask=cfg.min_mask,
        max_mask=cfg.max_mask,
        epochs=cfg.epochs,
        device=cfg.device,
    )

    eddi_selector = Ma2018AFAMethod(pv, classifier, num_classes=d_out)
    pretrained_model_artifact = wandb.Artifact(
        name=f"pretrain_ma2018-{cfg.dataset_artifact_name.split(':')[0]}",
        type="pretrained_model",
    )
    with TemporaryDirectory(delete=False) as tmp_path_str:
        tmp_path = Path(tmp_path_str)
        eddi_selector.save(tmp_path)
        del eddi_selector
        eddi_selector = Ma2018AFAMethod.load(tmp_path, device=device)

    pretrained_model_artifact.add_file(str(tmp_path / "model.pt"))
    run.log_artifact(
        pretrained_model_artifact,
        aliases=[*cfg.output_artifact_aliases, datetime.now().strftime("%b%d")],
    )
    run.finish()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
