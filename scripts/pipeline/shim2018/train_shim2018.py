import argparse
import os
import subprocess
import time

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, nargs="+")
    parser.add_argument("--split", type=int, required=True, nargs="+")
    parser.add_argument("--budgets", type=str, required=True, nargs="+")
    parser.add_argument("--launcher", default="custom_slurm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pretrain-alias", type=str, required=True)
    parser.add_argument("--output-alias", type=str, required=True)
    parser.add_argument("--wandb-entity", type=str, default="afa-team")
    parser.add_argument("--wandb-project", type=str, default="afa-benchmark")
    parser.add_argument("--min-acquisition-cost", type=float, required=True)
    parser.add_argument("--max-acquisition-cost", type=float, required=True)
    parser.add_argument("--n-acquisition-cost", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["WANDB_ENTITY"] = args.wandb_entity
    os.environ["WANDB_PROJECT"] = args.wandb_project

    print("Starting shim2018 training jobs...")
    time.sleep(1)

    # Generate acquisition costs uniformly in the specified range
    acquisition_costs = np.linspace(
        args.min_acquisition_cost,
        args.max_acquisition_cost,
        args.n_acquisition_cost,
    )
    acquisition_costs_str = ",".join(
        f"{cost:.6f}" for cost in acquisition_costs
    )

    jobs = []
    for dataset, budgets_str in zip(args.dataset, args.budgets, strict=False):
        pretrained_model_artifact_names = [
            f"pretrain_shim2018-{dataset}_split_{split}:{args.pretrain_alias}"
            for split in args.split
        ]
        cmd = (
            f"uv run scripts/train_methods/train_shim2018.py -m "
            f'output_artifact_aliases=["{args.output_alias}"] '
            f"dataset@_global_={dataset} "
            f"pretrained_model_artifact_name={','.join(pretrained_model_artifact_names)} "
            f'hard_budget="{budgets_str}" '
            f"acquisition_cost={acquisition_costs_str} "
            f"device={args.device} hydra/launcher={args.launcher}"
        )
        jobs.append(cmd)

    subprocess.run(["mprocs"] + jobs, check=False)


if __name__ == "__main__":
    main()
