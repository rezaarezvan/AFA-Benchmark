import torch
import yaml
import argparse
import torch.nn as nn
from pathlib import Path
from afa_discriminative.utils import MaskLayer
from afa_discriminative.models import fc_Net
from afa_discriminative.datasets import prepare_datasets
from afa_discriminative.afa_methods import GreedyDynamicSelection, Covert2023AFAMethod
from common.utils import dict_to_namespace, set_seed
from common.custom_types import AFADataset
from common.registry import AFA_DATASET_REGISTRY


def main(
    pretrain_config_path: Path,
    train_config_path: Path,
    dataset_type: str,
    train_dataset_path: Path,
    val_dataset_path: Path,
    pretrained_model_path: Path,
    hard_budget: int,
    seed: int,
    afa_method_path: Path,
):
    set_seed(seed)
    with open(train_config_path, "r") as file:
        train_config_dict: dict = yaml.safe_load(file)
    train_config = dict_to_namespace(train_config_dict)

    with open(pretrain_config_path, "r") as file:
        pretrain_config_dict: dict = yaml.safe_load(file)
    pretrain_config = dict_to_namespace(pretrain_config_dict)

    device = torch.device(train_config.device)

    train_dataset: AFADataset = AFA_DATASET_REGISTRY[dataset_type].load(
        train_dataset_path
    )
    val_dataset: AFADataset = AFA_DATASET_REGISTRY[dataset_type].load(val_dataset_path)
    train_loader, val_loader, d_in, d_out = prepare_datasets(
        train_dataset, val_dataset, pretrain_config.batch_size
    )

    predictor = fc_Net(
        input_dim=d_in * 2,
        output_dim=d_out,
        hidden_layer_num=len(pretrain_config.architecture.hidden_units),
        hidden_unit=pretrain_config.architecture.hidden_units,
        activations=pretrain_config.architecture.activations,
        drop_out_rate=pretrain_config.architecture.dropout,
        flag_drop_out=pretrain_config.architecture.flag_drop_out,
        flag_only_output_layer=pretrain_config.architecture.flag_only_output_layer,
    )

    predictor.load_state_dict(
        torch.load(pretrained_model_path / "model.pt", map_location=device)[
            "predictor_state_dict"
        ]
    )

    selector = fc_Net(
        input_dim=d_in * 2,
        output_dim=d_in,
        hidden_layer_num=len(train_config.architecture.hidden_units),
        hidden_unit=train_config.architecture.hidden_units,
        activations=train_config.architecture.activations,
        drop_out_rate=train_config.architecture.dropout,
        flag_drop_out=train_config.architecture.flag_drop_out,
        flag_only_output_layer=train_config.architecture.flag_only_output_layer,
    )

    mask_layer = MaskLayer(append=True)
    gdfs = GreedyDynamicSelection(selector, predictor, mask_layer).to(device)
    gdfs.fit(
        train_loader,
        val_loader,
        lr=train_config.lr,
        nepochs=train_config.nepochs,
        max_features=hard_budget,
        loss_fn=nn.CrossEntropyLoss(),
        patience=train_config.patience,
        verbose=True,
    )

    afa_method = Covert2023AFAMethod(gdfs.selector.cpu(), gdfs.predictor.cpu())
    # afa_method_path.mkdir(parents=True, exist_ok=True)
    afa_method.save(afa_method_path)
    with open(afa_method_path / "params.yml", "w") as file:
        yaml.dump(
            {
                "hard_budget": hard_budget,
                "seed": seed,
                "dataset_type": dataset_type,
                "train_dataset_path": str(train_dataset_path),
                "val_dataset_path": str(val_dataset_path),
                "pretrained_model_path": str(pretrained_model_path),
            },
            file,
        )

    print(f"Covert2023AFAMethod saved to {afa_method_path}")


if __name__ == "__main__":
    # Use argparse to choose config file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrain_config_path",
        type=Path,
        required=True,
        help="Path to YAML config file used for pretraining",
    )
    parser.add_argument(
        "--train_config_path",
        type=Path,
        required=True,
        help="Path to YAML config file for this training",
    )
    parser.add_argument(
        "--dataset_type", type=str, required=True, choices=AFA_DATASET_REGISTRY.keys()
    )
    parser.add_argument("--train_dataset_path", type=Path, required=True)
    parser.add_argument("--val_dataset_path", type=Path, required=True)
    parser.add_argument(
        "--pretrained_model_path",
        type=Path,
        required=True,
        help="Path to pretrained model folder",
    )
    parser.add_argument("--hard_budget", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--afa_method_path",
        type=Path,
        required=True,
        help="Path to folder to save the trained AFA method",
    )
    args = parser.parse_args()

    main(
        pretrain_config_path=args.pretrain_config_path,
        train_config_path=args.train_config_path,
        dataset_type=args.dataset_type,
        train_dataset_path=args.train_dataset_path,
        val_dataset_path=args.val_dataset_path,
        pretrained_model_path=args.pretrained_model_path,
        hard_budget=args.hard_budget,
        seed=args.seed,
        afa_method_path=args.afa_method_path,
    )
