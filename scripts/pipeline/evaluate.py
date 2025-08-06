import argparse
import os
import subprocess
from collections.abc import Sequence
import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--launcher", default="custom_slurm")
    parser.add_argument("--output-alias", default="tmp")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--yaml", required=True, help="YAML file with artifact names")
    return parser.parse_args()


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
    args = parse_args()
    os.environ["WANDB_ENTITY"] = args.wandb_entity
    os.environ["WANDB_PROJECT"] = args.wandb_project

    with open(args.yaml) as f:
        config = yaml.safe_load(f)

    jobs = []
    for dataset_type in config:
        for split_idx in config[dataset_type]:
            cmd = build_eval_job(
                config[dataset_type][split_idx]["trained_method_artifact_names"],
                config[dataset_type][split_idx]["trained_classifier_artifact_name"],
                args,
            )
            jobs.append(cmd)

    # Launch jobs with mprocs
    mprocs_cmd = ["mprocs"] + [" ".join(cmd) for cmd in jobs]
    subprocess.run(mprocs_cmd)


if __name__ == "__main__":
    main()
