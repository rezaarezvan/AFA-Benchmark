"""
Runs pretrain_shim2018 on all combinations of dataset_type, split and seed.
"""

import argparse
from pathlib import Path
import subprocess
from coolname import generate_slug
import yaml
from tqdm import tqdm
import time
from collections import defaultdict
from typing import Callable, Container, NamedTuple

from common.utils import get_folders_with_matching_params


class PretrainJobConfig(NamedTuple):
    dataset_type: str
    split: int
    seed: int


class TrainJobConfig(NamedTuple):
    dataset_type: str
    split: int
    seed: int
    hard_budget: int


class JobConfig(NamedTuple):
    dataset_type: str
    split: int
    seed: int
    hard_budget: int | None  # only used for training jobs


def is_job_finished(job_id: str) -> bool:
    """Check if the job with job_id is finished."""
    result = subprocess.run(
        ["squeue", "--job", job_id],
        capture_output=True,
        text=True,
        check=False
    )
    # If the job ID is not found in the output, it means the job is finished
    return job_id not in result.stdout


def wait_for_jobs(job_ids: set[str]) -> None:
    """Wait for all jobs in job_ids to finish."""
    start_time = time.time()
    completed_jobs = set()
    with tqdm(total=len(job_ids), desc="Jobs progress") as pbar:
        while len(job_ids - completed_jobs) > 0:
            elapsed_time = time.time() - start_time
            hours, remainder = divmod(int(elapsed_time), 3600)
            minutes, seconds = divmod(remainder, 60)
            pbar.set_description(
                f"Jobs progress (Elapsed: {hours:02}:{minutes:02}:{seconds:02})"
            )
            time.sleep(10)  # Sleep for 10 seconds
            for job_id in job_ids - completed_jobs:
                if is_job_finished(job_id):
                    completed_jobs.add(job_id)
                    pbar.update(1)


def get_suitable_pretrained_model(
    pretrained_model_folder: Path, train_job_config: JobConfig
) -> Path:
    """Find a suitable pretrained model for the given training job configuration."""
    # Find a pretrained model that has the same dataset_type, split and seed
    pretrained_model_folders = get_folders_with_matching_params(
        pretrained_model_folder,
        {
            "dataset_type": train_job_config.dataset_type,
            "train_dataset_path": f"data/{train_job_config.dataset_type}/train_split_{train_job_config.split}.pt",
            "val_dataset_path": f"data/{train_job_config.dataset_type}/val_split_{train_job_config.split}.pt",
            "seed": train_job_config.seed,
        },
    )
    # There should be exactly one pretrained model
    assert len(pretrained_model_folders) == 1, (
        f"Found {len(pretrained_model_folders)} pretrained models for dataset_type={train_job_config.dataset_type}, split={train_job_config.split}, seed={train_job_config.seed} at {pretrained_model_folder}. Expected 1."
    )
    pretrained_model_path = pretrained_model_folders[0]

    return pretrained_model_path


def submit_job(
    job_config: JobConfig,
    env_vars: str,
    job_path: Path,
    job_type: str,
) -> str:
    result = subprocess.run(
        ["sbatch", f"--export={env_vars}", job_path],
        capture_output=True,
        text=True,
        check=True,
    )
    output = result.stdout.strip()
    job_id = output.split()[-1]
    print(
        f"Submitted {job_type} job with ID {job_id} for dataset_type={job_config.dataset_type}, split={job_config.split}, seed={job_config.seed}"
        + (
            f", hard_budget={job_config.hard_budget}"
            if hasattr(job_config, "hard_budget")
            else ""
        )
    )
    return job_id


def process_jobs(
    job_configs: set[JobConfig],
    job_config_to_slug: dict[JobConfig, str],
    job_config_to_id: dict[JobConfig, str],
    job_path: Path,
    job_type: str,
    generate_env_vars: Callable[[JobConfig, str, Path], str],
    status_folder: Path,
) -> None:
    """Process a set of jobs by submitting them to SLURM, waiting for completion, and retrying failed jobs.

    Args:
        job_configs (set): Set of job configurations.
        job_config_to_slug (dict): Mapping of job configurations to unique slugs.
        job_config_to_id (dict): Mapping of job configurations to SLURM job IDs.
        job_path (Path): Path to the job script.
        job_type (str): Type of the job (e.g., "pretraining" or "training").
        generate_env_vars (callable): Function to generate environment variables for a job configuration.
        status_folder (Path): Folder where status files are stored.

    """
    remaining_job_configs = set(job_configs)

    while remaining_job_configs:
        for job_config in remaining_job_configs:
            slug = job_config_to_slug[job_config]
            status_file = status_folder / slug / "status.txt"
            env_vars = generate_env_vars(job_config, slug, status_file)
            job_id = submit_job(job_config, env_vars, job_path, job_type)
            job_config_to_id[job_config] = job_id

        wait_for_jobs(
            {job_config_to_id[job_config] for job_config in remaining_job_configs}
        )

        completed_jobs = set()
        for job_config in remaining_job_configs:
            status_file = status_folder / job_config_to_slug[job_config] / "status.txt"
            job_id = job_config_to_id[job_config]
            if status_file.exists() and status_file.read_text().strip() == "success":
                completed_jobs.add(job_config)
            else:
                print(
                    f"Job {job_id} for dataset_type={job_config.dataset_type}, split={job_config.split}, seed={job_config.seed}"
                    + (
                        f", hard_budget={job_config.hard_budget}"
                        if hasattr(job_config, "hard_budget")
                        else ""
                    )
                    + " failed. Retrying."
                )
        remaining_job_configs -= completed_jobs


def main(
    dataset_types: list[str],
    splits: list[int],
    seeds: list[int],
    hard_budgets: dict[str, list[int]],
    pretrained_model_folder: Path,
    pretrain: bool,
    train: bool,
    pretrain_config_path: Path,
    train_config_path: Path,
    method_folder: Path,
    pretrain_job_path: Path,
    train_job_path: Path,
) -> None:
    """Run pretraining and training jobs on SLURM for a given set of parameters.

    Args:
        dataset_types (list[str]): List of dataset types. Each element should be a key in AFA_DATASET_REGISTRY.
        splits (list[int]): List of splits.
        seeds (list[int]): List of seeds.
        hard_budgets (dict[str, list[int]]): Dictionary mapping dataset types to hard budgets.
        pretrained_model_folder (Path): Path to the folder where pretrained models will be stored.
        pretrain (bool): Whether to run pretraining.
        train (bool): Whether to run training.
        pretrain_config_path (Path): Path to the pretrain configuration file.
        train_config_path (Path): Path to the train configuration file.
        method_folder (Path): Path to the folder where trained methods will be stored.
        pretrain_job_path (Path): Path to the pretrain job script.
        train_job_path (Path): Path to the train job script.

    """
    if pretrain:
        pretrain_job_configs: set[JobConfig] = {
            JobConfig(
                dataset_type=dataset_type, split=split, seed=seed, hard_budget=None
            )
            for dataset_type in dataset_types
            for split in splits
            for seed in seeds
        }
        pretrain_job_config_to_slug: dict[JobConfig, str] = defaultdict(
            lambda: generate_slug(2)
        )
        pretrain_job_config_to_id: dict[JobConfig, str] = {}

        def generate_pretrain_env_vars(
            job_config: JobConfig, slug: str, status_file: Path
        ) -> str:
            return (
                f"pretrain_config_path={pretrain_config_path},"
                f"dataset_type={job_config.dataset_type},"
                f"train_dataset_path=data/{job_config.dataset_type}/train_split_{job_config.split}.pt,"
                f"val_dataset_path=data/{job_config.dataset_type}/val_split_{job_config.split}.pt,"
                f"pretrained_model_path={pretrained_model_folder / slug},"
                f"seed={job_config.seed},"
                f"status_file={status_file}"
            )

        process_jobs(
            pretrain_job_configs,
            pretrain_job_config_to_slug,
            pretrain_job_config_to_id,
            pretrain_job_path,
            "pretraining",
            generate_pretrain_env_vars,
            pretrained_model_folder,
        )

        print("Pretraining finished, now starting training.")

    if train:
        train_job_configs: set[JobConfig] = {
            JobConfig(
                dataset_type=dataset_type,
                split=split,
                seed=seed,
                hard_budget=hard_budget,
            )
            for dataset_type in dataset_types
            for split in splits
            for seed in seeds
            for hard_budget in hard_budgets[dataset_type]
        }
        train_job_config_to_slug: dict[JobConfig, str] = defaultdict(
            lambda: generate_slug(2)
        )
        train_job_config_to_id: dict[JobConfig, str] = {}

        def generate_train_env_vars(
            job_config: JobConfig, slug: str, status_file: Path
        ) -> str:
            pretrained_model_path = get_suitable_pretrained_model(
                pretrained_model_folder, job_config
            )
            return (
                f"pretrain_config={pretrain_config_path},"
                f"train_config={train_config_path},"
                f"dataset_type={job_config.dataset_type},"
                f"train_dataset_path=data/{job_config.dataset_type}/train_split_{job_config.split}.pt,"
                f"val_dataset_path=data/{job_config.dataset_type}/val_split_{job_config.split}.pt,"
                f"pretrained_model_path={pretrained_model_path},"
                f"hard_budget={job_config.hard_budget},"
                f"seed={job_config.seed},"
                f"afa_method_path={method_folder / slug},"
                f"status_file={status_file}"
            )

        process_jobs(
            train_job_configs,
            train_job_config_to_slug,
            train_job_config_to_id,
            train_job_path,
            "training",
            generate_train_env_vars,
            method_folder,
        )

        print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Toggle pretraining and training as flags
    parser.add_argument(
        "--no-pretrain",
        action="store_true",
        help="Don't run pretraining",
        default=False,
    )
    parser.add_argument(
        "--no-train", action="store_true", help="Don't run training", default=False
    )
    timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    parser.add_argument(
        "--pretrained_model_folder",
        type=Path,
        default=Path(f"models/pretrained/shim2018/{timestr}"),
        help="Path to the folder where pretrained models will be stored (or are already stored)",
    )
    parser.add_argument(
        "--method_folder",
        type=Path,
        default=Path(f"models/methods/shim2018/{timestr}"),
        help="Path to the folder where trained methods will be stored",
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
    assert args.pipeline_config_path.exists(), (
        "Slurm pipeline config path does not exist"
    )

    # Load pipeline config
    with open(args.pipeline_config_path) as config_file:
        config = yaml.safe_load(config_file)

    main(
        dataset_types=config["dataset_types"],
        splits=config["dataset_splits"],
        seeds=config["seeds"],
        hard_budgets=config["hard_budgets"],
        pretrained_model_folder=args.pretrained_model_folder,
        pretrain=not args.no_pretrain,
        train=not args.no_train,
        pretrain_config_path=args.pretrain_config_path,
        train_config_path=args.train_config_path,
        method_folder=args.method_folder,
        pretrain_job_path=config["pretrain_job_path"],
        train_job_path=config["train_job_path"],
    )
