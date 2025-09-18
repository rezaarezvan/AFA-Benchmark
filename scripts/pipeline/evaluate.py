import argparse
import os
import subprocess
from collections.abc import Sequence

import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--launcher",
        default="basic",
        type=str,
        help='Train locally in sequence or in parallel using Slurm. Value should be "basic" or one of the files (without suffix) defined in conf/global/hydra/launcher/',
    )
    parser.add_argument(
        "--output-alias",
        type=str,
        required=True,
        help="The alias that the resulting evaluation artifact should have.",
    )
    parser.add_argument("--wandb-entity", type=str)
    parser.add_argument("--wandb-project", type=str)
    parser.add_argument(
        "--yaml",
        type=str,
        required=True,
        help="Path to YAML file with artifact names",
    )
    args, unknown = parser.parse_known_args()
    return args, unknown


def build_eval_job(
    trained_method_artifact_names: Sequence[str],
    trained_classifier_artifact_name: str,
    args: argparse.Namespace,
):
    return [
        "uv",
        "run",
        "scripts/evaluation/eval_afa_method.py",
        "-m",
        f'output_artifact_aliases=["{args.output_alias}"]',
        f"trained_method_artifact_name={','.join(trained_method_artifact_names)}",
        f"trained_classifier_artifact_name={trained_classifier_artifact_name},null",
        f"batch_size={args.batch_size}",
        f"device={args.device}",
        f"hydra/launcher={args.launcher}",
    ]


def main():
    args, extra_args = parse_args()
    extra_args_str = " ".join(extra_args)
    if args.wandb_entity is not None:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    if args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    with open(args.yaml) as f:
        config = yaml.safe_load(f)

    jobs = []
    for dataset_type in config:
        for split_idx in config[dataset_type]:
            jobs.append(
                [
                    "uv",
                    "run",
                    "scripts/evaluation/eval_afa_method.py",
                    "-m",
                    f'output_artifact_aliases=["{args.output_alias}"]',
                    f"trained_method_artifact_name={','.join(config[dataset_type][split_idx]['trained_method_artifact_names'])}",
                    f"trained_classifier_artifact_name={config[dataset_type][split_idx]['trained_classifier_artifact_name']},null",
                    f"batch_size={args.batch_size}",
                    f"device={args.device}",
                    f"hydra/launcher={args.launcher}",
                    f"{extra_args_str}",
                ]
            )

    # Launch jobs with mprocs
    mprocs_cmd = ["mprocs"] + [" ".join(cmd) for cmd in jobs]
    subprocess.run(mprocs_cmd, check=False)


if __name__ == "__main__":
    main()
