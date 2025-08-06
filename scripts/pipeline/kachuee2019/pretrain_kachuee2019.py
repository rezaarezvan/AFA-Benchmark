import argparse
import os
import subprocess
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, nargs="+")
    parser.add_argument("--split", type=int, required=True, nargs="+")
    parser.add_argument("--launcher", default="custom_slurm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-alias", type=str, required=True)
    parser.add_argument("--output-alias", type=str, required=True)
    parser.add_argument("--wandb-entity", type=str, default="afa-team")
    parser.add_argument("--wandb-project", type=str, default="afa-benchmark")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["WANDB_ENTITY"] = args.wandb_entity
    os.environ["WANDB_PROJECT"] = args.wandb_project

    print("Starting kachuee2019 pretraining jobs...")
    time.sleep(1)

    jobs = []
    for dataset in args.dataset:
        dataset_artifact_names = [
            f"{dataset}_split_{split}:{args.dataset_alias}" for split in args.split
        ]
        cmd = (
            f"uv run scripts/pretrain_models/pretrain_kachuee2019.py -m "
            f'output_artifact_aliases=["{args.output_alias}"] '
            f"dataset@_global_={dataset} "
            f"dataset_artifact_name={','.join(dataset_artifact_names)} "
            f"device={args.device} hydra/launcher={args.launcher}"
        )
        jobs.append(cmd)

    subprocess.run(["mprocs"] + jobs, check=False)


if __name__ == "__main__":
    main()
