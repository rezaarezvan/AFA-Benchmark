"""
Runs pretrain_shim2018 and train_shim2018 locally and synchronously on all combinations of dataset_type, split, and seed.
"""

import argparse
from pathlib import Path
import subprocess
from coolname import generate_slug
import torch
import yaml
from tqdm import tqdm
import time
from afa_rl.scripts.pretrain_shim2018 import main as pretrain_main
from afa_rl.scripts.train_shim2018 import main as train_main
from common.utils import get_folders_with_matching_params
from rich import print as rprint

def main(dataset_types: list[str], splits: list[int], seeds: list[int], hard_budgets: dict[str, list[int]], pretrained_model_folder: Path, pretrain: bool, train: bool, pretrain_config_path: Path, train_config_path: Path, method_folder: Path) -> None:
    if pretrain:
        for dataset_type in dataset_types:
            for split in splits:
                for seed in seeds:
                    slug = generate_slug(2)
                    rprint(f"[bold blue]Memory used before script: {torch.cuda.memory_allocated() / 1024**2:.2f} MB[/bold blue]")
                    rprint(f"[bold green]Pretraining with dataset_type={dataset_type}, split={split}, seed={seed}[/bold green]")
                    pretrain_main(
                        pretrain_config_path=pretrain_config_path,
                        dataset_type=dataset_type,
                        train_dataset_path=Path(f"data/{dataset_type}/train_split_{split}.pt"),
                        val_dataset_path=Path(f"data/{dataset_type}/val_split_{split}.pt"),
                        pretrained_model_path=pretrained_model_folder / slug,
                        seed=seed
                    )
                    # Clear GPU memory
                    torch.cuda.empty_cache()
                    # Print how much memory is used
                    rprint(f"[bold blue]Memory used after script: {torch.cuda.memory_allocated() / 1024**2:.2f} MB[/bold blue]")

        rprint("[bold magenta]Pretraining finished, now starting training.[/bold magenta]")

    if train:
        for dataset_type in dataset_types:
            for split in splits:
                for seed in seeds:
                    # Find pretrained model that matches the dataset_type, split, and seed
                    pretrained_model_folders = get_folders_with_matching_params(folder=pretrained_model_folder, mapping={
                        "dataset_type": dataset_type,
                        "train_dataset_path": str(Path(f"data/{dataset_type}/train_split_{split}.pt")),
                        "val_dataset_path": str(Path(f"data/{dataset_type}/val_split_{split}.pt")),
                        "seed": seed,
                    })
                    assert len(pretrained_model_folders) == 1, f"Found {len(pretrained_model_folders)} pretrained model folders matching the parameters, expected 1."
                    pretrained_model_path = pretrained_model_folders[0]

                    for hard_budget in hard_budgets[dataset_type]:
                        slug = generate_slug(2)
                        rprint(f"[bold blue]Memory used before script: {torch.cuda.memory_allocated() / 1024**2:.2f} MB[/bold blue]")
                        rprint(f"[bold green]Training with dataset_type={dataset_type}, split={split}, seed={seed}, hard_budget={hard_budget}[/bold green]")
                        train_main(
                            pretrain_config_path=pretrain_config_path,
                            train_config_path=train_config_path,
                            dataset_type=dataset_type,
                            train_dataset_path=Path(f"data/{dataset_type}/train_split_{split}.pt"),
                            val_dataset_path=Path(f"data/{dataset_type}/val_split_{split}.pt"),
                            pretrained_model_path=pretrained_model_path,
                            hard_budget=hard_budget,
                            seed=seed,
                            afa_method_path=method_folder / slug,
                        )
                        # Clear GPU memory
                        torch.cuda.empty_cache()
                        rprint(f"[bold blue]Memory used after script: {torch.cuda.memory_allocated() / 1024**2:.2f} MB[/bold blue]")

        rprint("[bold magenta]Training finished.[/bold magenta]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Toggle pretraining and training as flags
    parser.add_argument(
        "--pretrain", action="store_true", help="Run pretraining", default=True,
    )
    parser.add_argument(
        "--train", action="store_true", help="Run training", default=True
    )
    timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    parser.add_argument(
        "--pretrained_model_folder",
        type=Path,
        default=Path(f"models/pretrained/shim2018/{timestr}"),
        help="Path to the folder containing pretrained models",
    )
    parser.add_argument(
        "--method_folder",
        type=Path,
        default=Path(f"models/methods/shim2018/{timestr}"),
        help="Path to the folder containing method models",
    )
    parser.add_argument(
        "--pretrain_config_path",
        type=Path,
        default=Path("configs/shim2018/pretrain_shim2018.yml"),
        help="Path to the pretrain configuration file",
    )
    parser.add_argument(
        "--train_config_path",
        type=Path,
        default=Path("configs/shim2018/train_shim2018.yml"),
        help="Path to the train configuration file",
    )
    parser.add_argument(
        "--pipeline_config_path",
        type=Path,
        default=Path("configs/shim2018/pipeline.yml"),
        help="Path to the slurm pipeline configuration file",
    )
    args = parser.parse_args()

    # Validate paths
    assert args.pretrain_config_path.exists(), "Pretrain config path does not exist"
    assert args.train_config_path.exists(), "Train config path does not exist"
    assert args.pipeline_config_path.exists(), "Slurm pipeline config path does not exist"

    # Load pipeline config
    with open(args.pipeline_config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    main(
        dataset_types=config["dataset_types"],
        splits=config["dataset_splits"],
        seeds=config["seeds"],
        hard_budgets=config["hard_budgets"],
        pretrained_model_folder=args.pretrained_model_folder,
        pretrain=args.pretrain,
        train=args.train,
        pretrain_config_path=args.pretrain_config_path,
        train_config_path=args.train_config_path,
        method_folder=args.method_folder
    )
