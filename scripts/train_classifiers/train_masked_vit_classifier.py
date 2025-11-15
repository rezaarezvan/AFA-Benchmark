import torch
import wandb
import timm
import hydra
import logging

from torch import nn
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tempfile import NamedTemporaryFile

from afabench.afa_discriminative.utils import MaskLayer2d
from afabench.common.classifiers import WrappedMaskedViTClassifier
from afabench.common.models import MaskedViTClassifier, MaskedViTTrainer
from afabench.common.config_classes import TrainMaskedViTClassifierConfig

from afabench.common.utils import (
    load_dataset_artifact,
    set_seed,
)

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/classifiers/masked_vit_classifier",
    config_name="config",
)
def main(cfg: TrainMaskedViTClassifierConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")

    run = wandb.init(
        group="train_masked_vit_classifier",
        job_type="train_classifier",
        # pyright: ignore[reportArgumentType]
        config=OmegaConf.to_container(cfg, resolve=True),
        dir="extra/wandb",
    )
    log.info(f"W&B run initialized: {run.name} ({run.id})")
    log.info(f"W&B run URL: {run.url}")

    train_dataset, val_dataset, _, dataset_metadata = load_dataset_artifact(
        cfg.dataset_artifact_name
    )
    d_out = train_dataset.n_classes
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,  # type: ignore
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=True,  # type: ignore
    )
    backbone = timm.create_model(cfg.model_name, pretrained=True)
    model = MaskedViTClassifier(backbone=backbone, num_classes=d_out)

    assert cfg.image_size % cfg.patch_size == 0
    mask_width = cfg.image_size // cfg.patch_size
    mask_layer = MaskLayer2d(
        mask_width=mask_width, patch_size=cfg.patch_size, append=False
    )
    trainer = MaskedViTTrainer(model, mask_layer).to(cfg.device)

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        lr=cfg.lr,
        nepochs=cfg.epochs,
        loss_fn=nn.CrossEntropyLoss(),
        val_loss_fn=nn.CrossEntropyLoss(),
        val_loss_mode="min",
        patience=cfg.patience,
        min_lr=cfg.min_lr,
        min_mask=cfg.min_masking_probability,
        max_mask=cfg.max_masking_probability,
    )

    wrapped_classifier = WrappedMaskedViTClassifier(
        module=model,
        device=torch.device(cfg.device),
        pretrained_model_name=cfg.model_name,
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
    )

    with NamedTemporaryFile(delete=False) as tmp_file:
        save_path = Path(tmp_file.name)
        wrapped_classifier.save(save_path)

    trained_classifier_artifact = wandb.Artifact(
        name=f"masked_vit_classifier-{
            cfg.dataset_artifact_name.split(':')[0]
        }",
        type="trained_classifier",
        metadata={
            "dataset_type": dataset_metadata["dataset_type"],
            "seed": cfg.seed,
            "classifier_class_name": wrapped_classifier.__class__.__name__,
            "classifier_type": "MaskedViTClassifier",
            "pretrained_model_name": cfg.model_name,
            "image_size": cfg.image_size,
            "patch_size": cfg.patch_size,
        },
    )
    trained_classifier_artifact.add_file(str(save_path), name="classifier.pt")
    run.log_artifact(
        trained_classifier_artifact, aliases=cfg.output_artifact_aliases
    )
    run.finish()


if __name__ == "__main__":
    main()
