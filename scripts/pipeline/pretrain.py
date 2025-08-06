import argparse
import os
import subprocess
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method-name",
        type=str,
        required=True,
        help='Determines what sort of model gets pretrained. Example: "kachuee2019" or "shim2018".',
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        nargs="+",
        help='Which dataset(s) the model gets pretrained on. Example: "cube AFAContext"',
    )
    parser.add_argument(
        "--split",
        type=int,
        required=True,
        nargs="+",
        help='Which dataset splits to use. Example: "1 2 3"',
    )
    parser.add_argument(
        "--launcher",
        default="custom_slurm",
        help='Train locally in sequence or in parallel using Slurm. Value should be "submitit_basic" or one of the files (without suffix) defined in conf/global/hydra/launcher/',
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--dataset-alias",
        type=str,
        required=True,
        help="The alias that was specified when generating data (generate_dataset.py)",
    )
    parser.add_argument(
        "--output-alias",
        type=str,
        required=True,
        help="The alias that the resulting pretrained model should have.",
    )
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-project", type=str, default="")
    args, unknown = parser.parse_known_args()
    return args, unknown


def main():
    args, extra_args = parse_args()
    extra_args_str = " ".join(extra_args)
    os.environ["WANDB_ENTITY"] = args.wandb_entity
    os.environ["WANDB_PROJECT"] = args.wandb_project

    print("Starting pretraining jobs...")
    time.sleep(1)

    jobs = []
    for dataset in args.dataset:
        dataset_artifact_names = [
            f"{dataset}_split_{split}:{args.dataset_alias}" for split in args.split
        ]
        cmd = (
            f"uv run scripts/pretrain_models/pretrain_{args.method_name}.py -m "
            f'output_artifact_aliases=["{args.output_alias}"] '
            f"dataset@_global_={dataset} "
            f"dataset_artifact_name={','.join(dataset_artifact_names)} "
            f"device={args.device} hydra/launcher={args.launcher} "
            f"{extra_args_str}"
        )
        jobs.append(cmd)

    subprocess.run(["mprocs"] + jobs)


if __name__ == "__main__":
    main()
