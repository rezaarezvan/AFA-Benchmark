import itertools
import subprocess

BUDGETS = {
    'cube': 10,
    'AFAContext': 10,
    'MNIST': 30,
    'FashionMNIST': 30,
    'diabetes': 15,
    'miniboone': 15,
    'physionet': 15,
}
SPLITS = [1, 2, 3, 4, 5]
ALIAS = "latest"


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Launch AACO evaluation jobs on SLURM")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running")
    args = parser.parse_args()

    print("=== Launching AACO Evaluation Jobs ===")

    for dataset, split in itertools.product(BUDGETS.keys(), SPLITS):
        cmd = f"""uv run scripts/evaluation/eval_afa_method.py -m \
trained_method_artifact_name=aaco-{dataset}_split_{split}:{ALIAS} \
trained_classifier_artifact_name=null \
+budget={BUDGETS[dataset]} \
dataset_split=validation \
output_artifact_aliases=["{ALIAS}"] \
hydra/launcher=custom_slurm"""

        print(f"Submitting: {dataset}_split_{split}_budget_{BUDGETS[dataset]}")
        if not args.dry_run:
            subprocess.Popen(cmd, shell=True)  # Non-blocking

    print("All evaluation jobs submitted to SLURM")


if __name__ == "__main__":
    main()
