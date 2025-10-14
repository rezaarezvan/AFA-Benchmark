import gc
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import hydra
import torch
from dacite import from_dict
from omegaconf import OmegaConf
from torch import nn
from torchmetrics import Accuracy

import wandb
from afa_discriminative.afa_methods import CMIEstimator, Gadgil2023AFAMethod
from afa_discriminative.datasets import prepare_datasets
from afa_discriminative.models import (
    ConvNet,
    Predictor,
    ResNet18Backbone,
    resnet18,
)
from afa_discriminative.utils import MaskLayer2d
from common.config_classes import (
    Gadgil2023Pretraining2DConfig,
    Gadgil2023Training2DConfig,
)
from common.utils import (
    get_class_probabilities,
    load_dataset_artifact,
    set_seed,
)

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../conf/train/gadgil2023",
    config_name="config",
)
def main(cfg: Gadgil2023Training2DConfig):
    log.debug(cfg)
    print(OmegaConf.to_yaml(cfg))
    run = wandb.init(
        config=cast(
            "dict[str, Any]", OmegaConf.to_container(cfg, resolve=True)
        ),
        job_type="training",
        tags=["DIME"],
        dir="wandb",
    )
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    pretrained_model_artifact = wandb.use_artifact(
        cfg.pretrained_model_artifact_name, type="pretrained_model"
    )
    pretrained_model_artifact_dir = Path(pretrained_model_artifact.download())
    artifact_filenames = [
        f.name for f in pretrained_model_artifact_dir.iterdir()
    ]
    assert {"model.pt"}.issubset(artifact_filenames), (
        f"Dataset artifact must contain a model.pt file. Instead found: {artifact_filenames}"
    )
    pretraining_run = pretrained_model_artifact.logged_by()
    assert pretraining_run is not None, (
        "Pretrained model artifact must be logged by a run."
    )
    pretrained_model_config_dict = pretraining_run.config
    pretrained_model_config: Gadgil2023Pretraining2DConfig = from_dict(
        data_class=Gadgil2023Pretraining2DConfig,
        data=pretrained_model_config_dict,
    )
    train_dataset, val_dataset, test_dataset, dataset_metadata = (
        load_dataset_artifact(pretrained_model_config.dataset_artifact_name)
    )
    train_class_probabilities = get_class_probabilities(train_dataset.labels)
    class_weights = len(train_class_probabilities) / (
        len(train_class_probabilities) * train_class_probabilities
    )
    class_weights = class_weights.to(device)
    train_loader, val_loader, d_in, d_out = prepare_datasets(
        train_dataset, val_dataset, cfg.batch_size
    )

    base = resnet18(pretrained=True)
    backbone, expansion = ResNet18Backbone(base)
    predictor = Predictor(backbone, expansion, num_classes=d_out).to(device)

    checkpoint = torch.load(
        str(pretrained_model_artifact_dir / "model.pt"),
        map_location=device,
        weights_only=False,
    )
    predictor.load_state_dict(checkpoint["predictor_state_dict"])

    arch = checkpoint["architecture"]
    image_size = arch["image_size"]
    patch_size = arch["patch_size"]
    assert image_size % patch_size == 0
    mask_width = arch["mask_width"]
    n_patches = int(mask_width) ** 2

    value_network = ConvNet(backbone, expansion).to(device)

    mask_layer = MaskLayer2d(
        mask_width=mask_width, patch_size=patch_size, append=False
    )
    x0, _ = next(iter(train_loader))
    with torch.no_grad():
        logits0 = value_network(
            mask_layer(
                x0.to(device), torch.zeros(len(x0), n_patches, device=device)
            )
        )
    assert logits0.shape[1] == n_patches, (
        f"Value Network outputs {logits0.shape[1]} != n_patches {n_patches}"
    )

    greedy_cmi_estimator = CMIEstimator(
        value_network, predictor, mask_layer
    ).to(device)
    greedy_cmi_estimator.fit(
        train_loader,
        val_loader,
        lr=cfg.lr,
        min_lr=cfg.min_lr,
        nepochs=cfg.nepochs,
        max_features=cfg.hard_budget,
        eps=cfg.eps,
        loss_fn=nn.CrossEntropyLoss(reduction="none", weight=class_weights),
        # val_loss_fn=Accuracy(task="multiclass", num_classes=d_out).to(device),
        # val_loss_mode="max",
        val_loss_fn=None,
        val_loss_mode=None,
        eps_decay=cfg.eps_decay,
        eps_steps=cfg.eps_steps,
        patience=cfg.patience,
        feature_costs=None,
    )

    afa_method = Gadgil2023AFAMethod(
        greedy_cmi_estimator.value_network.cpu(),
        greedy_cmi_estimator.predictor.cpu(),
        modality="image",
        n_patches=n_patches,
    )
    afa_method.image_size = image_size
    afa_method.patch_size = patch_size
    afa_method.mask_width = mask_width

    with TemporaryDirectory(delete=False) as tmp_path_str:
        tmp_path = Path(tmp_path_str)
        afa_method.save(tmp_path)
        del afa_method
        afa_method = Gadgil2023AFAMethod.load(tmp_path, device=device)
        afa_method_artifact = wandb.Artifact(
            name=f"train_gadgil2023-{pretrained_model_config.dataset_artifact_name.split(':')[0]}-budget_{cfg.hard_budget}-seed_{cfg.seed}",
            type="trained_method",
            metadata={
                "method_type": "gadgil2023",
                "dataset_artifact_name": pretrained_model_config.dataset_artifact_name,
                "dataset_type": dataset_metadata["dataset_type"],
                "budget": cfg.hard_budget,
                "seed": cfg.seed,
            },
        )
        afa_method_artifact.add_file(str(tmp_path / "model.pt"))
        run.log_artifact(
            afa_method_artifact, aliases=cfg.output_artifact_aliases
        )

    run.finish()

    gc.collect()  # Force Python GC
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Release cached memory held by PyTorch CUDA allocator
        torch.cuda.synchronize()  # Optional, wait for CUDA ops to finish


if __name__ == "__main__":
    main()
