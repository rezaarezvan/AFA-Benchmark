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
        help='Determines what sort of method gets trained. Example: "kachuee2019" or "shim2018".',
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        nargs="+",
        help='Which dataset(s) the method gets trained on. Example: "cube AFAContext"',
    )
    parser.add_argument(
        "--dataset-alias",
        type=str,
        required=False,  # Now optional
        help="The alias that was specified when generating data (generate_dataset.py). Only required if the method has no pretraining stage.",
    )
    parser.add_argument(
        "--split",
        type=int,
        required=True,
        nargs="+",
        help='Which dataset splits to use. Example: "1 2 3"',
    )
    parser.add_argument(
        "--budgets",
        type=str,
        required=True,
        nargs="+",
        help='Which hard budgets to use. Example: "5 10 15"',
    )
    parser.add_argument(
        "--launcher",
        default="basic",
        help='Train locally in sequence or in parallel using Slurm. Value should be "basic" or one of the files (without suffix) defined in conf/global/hydra/launcher/',
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--pretrain-alias",
        type=str,
        required=False,
        help='The alias that was specified when pretraining the model ("--output-alias" of pretrain.py). Only required if the method has a pretraining stage.',
    )
    parser.add_argument(
        "--output-alias",
        type=str,
        required=True,
        help="The alias that the resulting trained method should have.",
    )
    parser.add_argument("--wandb-entity", type=str)
    parser.add_argument("--wandb-project", type=str)
    args, unknown = parser.parse_known_args()

    # Verify that either dataset-alias or pretrain-alias is provided, but not both
    if (args.dataset_alias is not None) and (args.pretrain_alias is not None):
        parser.error(
            "Provide either --dataset-alias or --pretrain-alias, but not both."
        )
    if (args.dataset_alias is None) and (args.pretrain_alias is None):
        parser.error(
            "You must provide either --dataset-alias or --pretrain-alias."
        )

    return args, unknown


def main():
    args, extra_args = parse_args()
    extra_args_str = " ".join(extra_args)
    if args.wandb_entity is not None:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    if args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    print("Starting training jobs...")
    time.sleep(1)

    jobs = []
    for dataset, budgets_str in zip(args.dataset, args.budgets, strict=False):
        if args.pretrain_alias:
            pretrained_model_artifact_names = [
                f"pretrain_{args.method_name}-{dataset}_split_{split}:{args.pretrain_alias}"
                for split in args.split
            ]
            pretrained_model_or_dataset_names = f"pretrained_model_artifact_name={','.join(pretrained_model_artifact_names)}"
        elif args.dataset_alias:
            dataset_artifact_names = [
                f"{dataset}_split_{split}:{args.dataset_alias}"
                for split in args.split
            ]
            pretrained_model_or_dataset_names = (
                f"dataset_artifact_name={','.join(dataset_artifact_names)}"
            )
        else:
            raise Exception("Unreachable")

        cmd = (
            f"uv run scripts/train_methods/train_{args.method_name}.py -m "
            f'output_artifact_aliases=["{args.output_alias}"] '
            f"dataset@_global_={dataset} "
            f"{pretrained_model_or_dataset_names} "
            f'hard_budget="{budgets_str}" '
            f"device={args.device} "
            f"hydra/launcher={args.launcher} "
            f"{extra_args_str}"
        )
        jobs.append(cmd)

    subprocess.run(["mprocs"] + jobs, check=False)


if __name__ == "__main__":
    main()
