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

from common.utils import get_folders_with_matching_params


def are_jobs_finished(job_ids: list[int]) -> bool:
    """Check if all jobs in job_ids are finished."""
    for job_id in job_ids:
        result = subprocess.run(
            ["squeue", "--job", str(job_id)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            print(f"Error checking job {job_id}: {result.stderr}")
            continue
        if str(job_id) in str(result.stdout):
            return False
    return True


def wait_for_jobs(job_ids: list[int]) -> None:
    """Wait for all jobs in job_ids to finish."""
    start_time = time.time()
    # Use a progress bar to visualize job completion
    with tqdm(total=len(job_ids), desc="Jobs progress") as pbar:
        completed_jobs = set()
        while len(completed_jobs) < len(job_ids):
            elapsed_time = time.time() - start_time
            hours, remainder = divmod(int(elapsed_time), 3600)
            minutes, seconds = divmod(remainder, 60)
            pbar.set_description(
                f"Jobs progress (Elapsed: {hours:02}:{minutes:02}:{seconds:02})"
            )
            time.sleep(10)  # Sleep for 10 seconds
            for job_id in job_ids:
                if job_id not in completed_jobs and are_jobs_finished([job_id]):
                    completed_jobs.add(job_id)
                    pbar.update(1)


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
        pretraining_job_ids = []

        for dataset_type in dataset_types:
            for split in splits:
                for seed in seeds:
                    slug = generate_slug(2)
                    env_vars: str = (
                        f"pretrain_config_path={pretrain_config_path},"
                        f"dataset_type={dataset_type},"
                        f"train_dataset_path=data/{dataset_type}/train_split_{split}.pt,"
                        f"val_dataset_path=data/{dataset_type}/val_split_{split}.pt,"
                        f"pretrained_model_path={pretrained_model_folder / slug},"
                        f"seed={seed}"
                    )
                    result = subprocess.run(
                        ["sbatch", f"--export={env_vars}", pretrain_job_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    assert result.returncode == 0, (
                        f"Error submitting job: {result.stderr}"
                    )
                    output = result.stdout.strip()
                    job_id = output.split()[-1]  # Extract the job ID from the output
                    print(
                        f"Submitted pretraining job with ID {job_id} for dataset_type={dataset_type}, split={split}, seed={seed}"
                    )
                    pretraining_job_ids.append(job_id)

        wait_for_jobs(pretraining_job_ids)

        print("Pretraining finished, now starting training.")

    if train:
        training_job_ids = []
        for dataset_type in dataset_types:
            for split in splits:
                for seed in seeds:
                    # Find a pretrained model that has the same dataset_type, split and seed
                    pretrained_model_folders = get_folders_with_matching_params(
                        pretrained_model_folder,
                        {
                            "dataset_type": dataset_type,
                            "train_dataset_path": f"data/{dataset_type}/train_split_{split}.pt",
                            "val_dataset_path": f"data/{dataset_type}/val_split_{split}.pt",
                            "seed": seed,
                        },
                    )
                    # There should be exactly one pretrained model
                    assert len(pretrained_model_folders) == 1, (
                        f"Found {len(pretrained_model_folders)} pretrained models for dataset_type={dataset_type}, split={split}, seed={seed} at {pretrained_model_folder}. Expected 1."
                    )
                    pretrained_model_path = pretrained_model_folders[0]

                    for hard_budget in hard_budgets[dataset_type]:
                        slug = generate_slug(2)
                        env_vars: str = (
                            f"pretrain_config={pretrain_config_path},"
                            f"train_config={train_config_path},"
                            f"dataset_type={dataset_type},"
                            f"train_dataset_path=data/{dataset_type}/train_split_{split}.pt,"
                            f"val_dataset_path=data/{dataset_type}/val_split_{split}.pt,"
                            f"pretrained_model_path={pretrained_model_path},"
                            f"hard_budget={hard_budget},"
                            f"seed={seed},"
                            f"afa_method_path={method_folder / slug}"
                        )
                        result = subprocess.run(
                            ["sbatch", f"--export={env_vars}", train_job_path],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                        )
                        assert result.returncode == 0, (
                            f"Error submitting job: {result.stderr}"
                        )
                        output = result.stdout.strip()
                        job_id = output.split()[
                            -1
                        ]  # Extract the job ID from the output
                        training_job_ids.append(job_id)
                        print(
                            f"Submitted training job with ID {job_id} for dataset_type={dataset_type}, split={split}, seed={seed}, hard_budget={hard_budget}"
                        )

        wait_for_jobs(training_job_ids)

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
    with open(args.pipeline_config_path, "r") as config_file:
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
