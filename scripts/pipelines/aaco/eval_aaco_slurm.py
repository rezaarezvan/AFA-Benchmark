import itertools
import subprocess

BUDGETS = {
    'cube': [3, 5, 10],
    'AFAContext': [3, 5, 10],
    'MNIST': [10, 20, 30],
    'FashionMNIST': [10, 20, 30],
    'diabetes': [15, 25, 35],
    'miniboone': [15, 25, 35],
    'physionet': [15, 25, 30]
}
SPLITS = [1, 2]
ALIAS = "latest"

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Launch AACO evaluation jobs on SLURM")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    print("=== Launching AACO Evaluation Jobs ===")

    for dataset, split in itertools.product(BUDGETS.keys(), SPLITS):
        for budget in BUDGETS[dataset]:
            cmd = f"""uv run scripts/evaluation/eval_afa_method.py \
trained_method_artifact_name=aaco-{dataset}_split_{split}:{ALIAS} \
trained_classifier_artifact_name=null \
+budget={budget} \
dataset_split=validation \
output_artifact_aliases=["{ALIAS}"] \
hydra/launcher=custom_slurm"""

            print(f"Submitting: {dataset}_split_{split}_budget_{budget}")
            if not args.dry_run:
                result = subprocess.run(cmd, shell=True)
                if result.returncode != 0:
                    print(f"Error submitting job for {dataset}_split_{split}_budget_{budget}")

    print("All evaluation jobs submitted to SLURM")

if __name__ == "__main__":
    main()
