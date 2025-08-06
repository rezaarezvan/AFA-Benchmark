import argparse
import os
import subprocess
import time
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train classifier with specified datasets and splits."
    )
    parser.add_argument("--dataset", type=str, required=True, nargs="+")
    parser.add_argument("--split", type=int, nargs="+", default=[1, 2])
    parser.add_argument(
        "--launcher",
        type=str,
        default="custom_slurm",
        help='Train locally in sequence or in parallel using Slurm. Value should be "basic" or one of the files (without suffix) defined in conf/global/hydra/launcher/',
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset-alias", type=str, default="tmp")
    parser.add_argument("--output-alias", type=str, default="tmp")
    parser.add_argument("--wandb-entity", type=str, default="afa-team")
    parser.add_argument("--wandb-project", type=str, default="afa-benchmark")
    return parser.parse_args()


def main():
    args = parse_args()

    os.environ["WANDB_ENTITY"] = args.wandb_entity
    os.environ["WANDB_PROJECT"] = args.wandb_project

    extra_opts = f"device={args.device} hydra/launcher={args.launcher}"

    print("Starting classifier training jobs...")
    time.sleep(1)

    jobs = []
    for dataset in args.dataset:
        dataset_artifact_names = [
            f"{dataset}_split_{split}:{args.dataset_alias}" for split in args.split
        ]
        cmd = (
            f"uv run scripts/train_classifiers/train_masked_mlp_classifier.py -m "
            f'output_artifact_aliases=["{args.output_alias}"] '
            f"dataset@_global_={dataset} "
            f"dataset_artifact_name={','.join(dataset_artifact_names)} "
            f"{extra_opts}"
        )
        jobs.append(cmd)

    subprocess.run(["mprocs"] + jobs)


if __name__ == "__main__":
    main()
