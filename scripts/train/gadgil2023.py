import gc
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import torch
from omegaconf import OmegaConf
from torch import nn
from torchrl.modules import MLP
from torchmetrics import Accuracy

from afabench import SAVE_PATH
from afabench.afa_discriminative.afa_methods import (
    CMIEstimator,
    Gadgil2023AFAMethod,
)
from afabench.afa_discriminative.datasets import prepare_datasets
from afabench.afa_discriminative.utils import MaskLayer
from afabench.common.config_classes import (
    Gadgil2023TrainingConfig,
)
from afabench.common.initializers.utils import get_afa_initializer_from_config
from afabench.common.utils import (
    get_class_frequencies,
    load_pretrained_model,
    load_dataset_splits,
    save_artifact,
    set_seed,
)

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train/gadgil2023",
    config_name="config",
)
def main(cfg: Gadgil2023TrainingConfig):
    log.debug(cfg)
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    (
        model_path,
        meta_data,
    ) = load_pretrained_model(
        Path(cfg.pretrained_path),
        device=device,
    )

    dataset_name = meta_data["dataset_name"]
    dataset_base_path = meta_data["dataset_base_path"]
    dataset_split_idx = meta_data["dataset_split_idx"]

    dataset_root = (
        Path(dataset_base_path) / dataset_name / str(dataset_split_idx)
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
        in_features = d_in * 2,
        out_features = d_out,
        num_cells = cfg.hidden_units,
        activation_class = torch.nn.ReLU,
        dropout = cfg.dropout,
    )

    # Load pretrained predictor weights
    ckpt = torch.load(model_path, map_location=device)
    predictor.load_state_dict(ckpt["predictor_state_dict"])

    value_network = MLP(
        in_features = d_in * 2,
        out_features = d_in,
        num_cells = cfg.hidden_units,
        activation_class = torch.nn.ReLU,
        dropout = cfg.dropout,
    )

    pred_linears = [m for m in predictor if isinstance(m, nn.Linear)]
    value_linears = [m for m in value_network if isinstance(m, nn.Linear)]

    for i in range(len(cfg.hidden_units)):
        value_linears[i].weight = pred_linears[i].weight
        value_linears[i].bias = pred_linears[i].bias

    mask_layer = MaskLayer(append=True)
    initializer = get_afa_initializer_from_config(cfg.initializer)
    initializer.set_seed(cfg.seed)

    greedy_cmi_estimator = CMIEstimator(
        value_network, predictor, mask_layer
    ).to(device)
    greedy_cmi_estimator.fit(
        train_loader,
        val_loader,
        lr=cfg.lr,
        nepochs=cfg.nepochs,
        max_features=cfg.hard_budget,
        eps=cfg.eps,
        loss_fn=nn.CrossEntropyLoss(reduction="none", weight=class_weights),
        val_loss_fn=Accuracy(task="multiclass", num_classes=d_out).to(device),
        val_loss_mode="max",
        eps_decay=cfg.eps_decay,
        eps_steps=cfg.eps_steps,
        patience=cfg.patience,
        feature_costs=train_dataset.get_feature_acquisition_costs().to(device),
        initializer=initializer,
    )

    afa_method = Gadgil2023AFAMethod(
        greedy_cmi_estimator.value_network.cpu(),
        greedy_cmi_estimator.predictor.cpu(),
        device=torch.device("cpu"),
        modality="tabular",
        d_in=d_in,
        d_out=d_out,
    )

    # Save locally
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        afa_method.save(tmp_path)

        artifact_identifier = f"{dataset_name.lower()}_split_{dataset_split_idx}_budget_{
            cfg.hard_budget
        }_seed_{cfg.seed}"
        artifact_dir = SAVE_PATH / artifact_identifier

        metadata_out = {
            "method_class_name": afa_method.__class__.__name__,
            "dataset_class_name": dataset_metadata["class_name"],
            "dataset_base_path": dataset_base_path,
            "budget": cfg.hard_budget,
            "seed": cfg.seed,
            "split_idx": dataset_split_idx,
            "initializer": cfg.initializer.class_name,
            "initializer_config": OmegaConf.to_container(cfg.initializer.kwargs),
        }

        save_artifact(
            artifact_dir=artifact_dir,
            files={f.name: f for f in tmp_path.iterdir() if f.is_file()},
            metadata=metadata_out,
        )

        log.info(f"Gadgil2023 method saved to: {artifact_dir}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
