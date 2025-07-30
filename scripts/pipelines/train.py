import argparse
import os
import subprocess
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-name", type=str, required=True, nargs="+")
    parser.add_argument("--dataset", type=str, required=True, nargs="+")
    parser.add_argument("--split", type=int, required=True, nargs="+")
    parser.add_argument("--budgets", type=str, required=True, nargs="+")
    parser.add_argument("--launcher", default="custom_slurm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pretrain-alias", type=str, required=True)
    parser.add_argument("--output-alias", type=str, required=True)
    parser.add_argument("--wandb-entity", type=str, default="afa-team")
    parser.add_argument("--wandb-project", type=str, default="afa-benchmark")
    args, unknown = parser.parse_known_args()
    return args, unknown


def main():
    args, extra_args = parse_args()
    extra_args_str = " ".join(extra_args)
    os.environ["WANDB_ENTITY"] = args.wandb_entity
    os.environ["WANDB_PROJECT"] = args.wandb_project

    print("Starting training jobs...")
    time.sleep(1)

    jobs = []
    for dataset, budgets_str in zip(args.dataset, args.budgets):
        pretrained_model_artifact_names = [
            f"pretrain_zannone2019-{dataset}_split_{split}:{args.pretrain_alias}"
            for split in args.split
        ]
        cmd = (
            f"uv run scripts/train_methods/train_{args.method_name}.py -m "
            f'output_artifact_aliases=["{args.output_alias}"] '
            f"dataset@_global_={dataset} "
            f"pretrained_model_artifact_name={','.join(pretrained_model_artifact_names)} "
            f'hard_budget="{budgets_str}" '
            f"device={args.device} hydra/launcher={args.launcher} "
            f"{extra_args_str}"
        )
        jobs.append(cmd)

    subprocess.run(["mprocs"] + jobs)


if __name__ == "__main__":
    main()
