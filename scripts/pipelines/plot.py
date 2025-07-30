import argparse
import os
import subprocess
import yaml
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-entity", default="afa-team")
    parser.add_argument("--wandb-project", default="afa-benchmark")
    parser.add_argument(
        "--yaml", required=True, help="YAML file with eval artifact names"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["WANDB_ENTITY"] = args.wandb_entity
    os.environ["WANDB_PROJECT"] = args.wandb_project

    print("Starting plotting job...")
    time.sleep(1)

    # with open(args.yaml) as f:
    #     config = yaml.safe_load(f)
    # eval_artifact_names = config["eval_artifact_names"]
    # joined_names = ",".join(eval_artifact_names)

    cmd = [
        "uv",
        "run",
        "scripts/plotting/plot_results.py",
        f"eval_artifact_config_path={args.yaml}",
    ]
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
