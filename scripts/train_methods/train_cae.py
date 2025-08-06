import gc
import wandb
import hydra
import logging
from pathlib import Path
from typing import Any, cast
from omegaconf import OmegaConf
import numpy as np
from tempfile import TemporaryDirectory
import torch
from torch import nn
from torchrl.modules import MLP
from torch.utils.data import DataLoader
from static.models import BaseModel
from static.utils import transform_dataset
from static.static_methods import DifferentiableSelector, ConcreteMask, StaticBaseMethod
from afa_discriminative.datasets import prepare_datasets
from common.utils import set_seed, load_dataset_artifact, get_class_probabilities
from common.config_classes import CAETrainingConfig


log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../conf/train/cae",
    config_name="config",
)
def main(cfg: CAETrainingConfig):
    log.debug(cfg)
    print(OmegaConf.to_yaml(cfg))
    run = wandb.init(
        config=cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True)),
        job_type="training",
        tags=["CAE"],
        dir="wandb",
    )
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    train_dataset, val_dataset, _, dataset_metadata = load_dataset_artifact(
        cfg.dataset_artifact_name
    )
    train_class_probabilities = get_class_probabilities(train_dataset.labels)
    class_weights = len(train_class_probabilities) / (
        len(train_class_probabilities) * train_class_probabilities
    )
    class_weights = class_weights.to(device)
    train_loader, val_loader, d_in, d_out = prepare_datasets(
        train_dataset, val_dataset, cfg.batch_size
    )

    model = MLP(
        in_features=d_in,
        out_features=d_out,
        num_cells=cfg.selector.num_cells,
        activation_class=nn.ReLU,
    )
    selector_layer = ConcreteMask(d_in, cfg.hard_budget)
    diff_selector = DifferentiableSelector(model, selector_layer).to(device)
    diff_selector.fit(
        train_loader,
        val_loader,
        lr=cfg.selector.lr,
        nepochs=cfg.selector.nepochs,
        loss_fn=nn.CrossEntropyLoss(weight=class_weights),
        patience=cfg.selector.patience,
        verbose=False,
    )

    logits = selector_layer.logits.cpu().data.numpy()
    ranked_features = np.sort(logits.argmax(axis=1))

    if len(np.unique(ranked_features)) != cfg.hard_budget:
        print(
            f"{len(np.unique(ranked_features))} selected instead of {cfg.hard_budget}, appending extras"
        )
    num_extras = cfg.hard_budget - len(np.unique(ranked_features))
    remaining_features = np.setdiff1d(np.arange(d_in), ranked_features)
    ranked_features = np.sort(
        np.concatenate([np.unique(ranked_features), remaining_features[:num_extras]])
    )

    predictors: dict[int, nn.Module] = {}
    selected_history: dict[int, list[int]] = {}

    num_features = list(range(1, cfg.hard_budget + 1))
    for num in num_features:
        selected_features = ranked_features[:num]
        selected_history[num] = selected_features.tolist()

        train_subset = transform_dataset(train_dataset, selected_features)
        val_subset = transform_dataset(val_dataset, selected_features)

        train_subset_loader = DataLoader(
            train_subset,
            batch_size=cfg.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
        val_subset_loader = DataLoader(
            val_subset, batch_size=cfg.batch_size, pin_memory=True
        )

        model = MLP(
            in_features=num,
            out_features=d_out,
            num_cells=cfg.classifier.num_cells,
            activation_class=nn.ReLU,
        )
        predictor = BaseModel(model).to(device)
        predictor.fit(
            train_subset_loader,
            val_subset_loader,
            lr=cfg.classifier.lr,
            nepochs=cfg.classifier.nepochs,
            loss_fn=nn.CrossEntropyLoss(weight=class_weights),
            verbose=False,
        )

        predictors[num] = model

    static_method = StaticBaseMethod(selected_history, predictors, device)

    with TemporaryDirectory(delete=False) as tmp_path_str:
        tmp_path = Path(tmp_path_str)
        static_method.save(tmp_path)
        del static_method
        static_method = StaticBaseMethod.load(tmp_path, device=device)
        static_method_artifact = wandb.Artifact(
            name=f"train_cae-{cfg.dataset_artifact_name.split(':')[0]}-budget_{cfg.hard_budget}-seed_{cfg.seed}",
            type="trained_method",
            metadata={
                "method_type": "cae",
                "dataset_artifact_name": cfg.dataset_artifact_name,
                "dataset_type": dataset_metadata["dataset_type"],
                "budget": cfg.hard_budget,
                "seed": cfg.seed,
            },
        )
        static_method_artifact.add_dir(str(tmp_path))
        run.log_artifact(static_method_artifact, aliases=cfg.output_artifact_aliases)

    run.finish()

    gc.collect()  # Force Python GC
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Release cached memory held by PyTorch CUDA allocator
        torch.cuda.synchronize()  # Optional, wait for CUDA ops to finish


if __name__ == "__main__":
    main()
