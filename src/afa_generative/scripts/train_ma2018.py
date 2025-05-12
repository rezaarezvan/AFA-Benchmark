import torch
import yaml
import argparse
import torch.nn as nn
from pathlib import Path
from afa_generative.afa_methods import Ma2018AFAMethod
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
    device = torch.device(train_config.device)
    afa_method: Ma2018AFAMethod = Ma2018AFAMethod.load(pretrained_model_path / "model.pt", device=device)
    afa_method_path.mkdir(parents=True, exist_ok=True)
    afa_method.save(afa_method_path / "model.pt")
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

    print(f"Ma2018AFAMethod saved to {afa_method_path}")


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
