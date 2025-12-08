import gc
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import torch
import wandb
from omegaconf import OmegaConf
from torch import nn
from torchrl.modules import MLP

from afabench import SAVE_PATH
from afabench.afa_discriminative.datasets import prepare_datasets
from afabench.afa_discriminative.models import MaskingPretrainer
from afabench.afa_discriminative.utils import MaskLayer
from afabench.common.config_classes import Covert2023PretrainingConfig
from afabench.common.utils import (
    get_class_frequencies,
    load_dataset_splits,
    save_artifact,
    set_seed,
)

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/pretrain/covert2023",
    config_name="config",
)
def main(cfg: Covert2023PretrainingConfig) -> None:
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")
    device = torch.device(cfg.device)

    dataset_root = (
        Path(cfg.dataset_base_path) / cfg.dataset_name / str(cfg.split_idx)
    )
    train_dataset, val_dataset, _, dataset_metadata = load_dataset_splits(
        dataset_root
    )

    _, train_labels = train_dataset.get_all_data()
    train_class_probabilities = get_class_frequencies(train_labels)
    class_weights = len(train_class_probabilities) / (
        len(train_class_probabilities) * train_class_probabilities
    )
    class_weights = class_weights.to(device)

    train_loader, val_loader, d_in, d_out = prepare_datasets(
        train_dataset, val_dataset, cfg.batch_size
    )

    predictor = MLP(
        in_features=d_in * 2,
        out_features=d_out,
        num_cells=cfg.hidden_units,
        activation_class=torch.nn.ReLU,
        dropout=cfg.dropout,
    )

    mask_layer = MaskLayer(append=True)
    pretrainer = MaskingPretrainer(predictor, mask_layer).to(device)

    pretrainer.fit(
        train_loader,
        val_loader,
        lr=cfg.lr,
        nepochs=cfg.nepochs,
        loss_fn=nn.CrossEntropyLoss(weight=class_weights),
        patience=cfg.patience,
        verbose=True,
        min_mask=cfg.min_masking_probability,
        max_mask=cfg.max_masking_probability,
    )

    artifact_dir = Path(cfg.output_dir) / cfg.dataset_name / str(cfg.split_idx)

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        torch.save(
            {"predictor_state_dict": predictor.state_dict()},
            tmp_path / "model.pt",
        )

        metadata = {
            "model_type": "Covert2023Predictor",
            "dataset_name": cfg.dataset_name,
            "dataset_base_path": cfg.dataset_base_path,
            "dataset_split_idx": cfg.split_idx,
            "seed": cfg.seed,
            "pretrain_config": OmegaConf.to_container(cfg),
        }

        save_artifact(
            artifact_dir=artifact_dir,
            files={"model.pt": tmp_path / "model.pt"},
            metadata=metadata,
        )

    log.info(f"Covert2023 pretrained model saved to: {artifact_dir}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
