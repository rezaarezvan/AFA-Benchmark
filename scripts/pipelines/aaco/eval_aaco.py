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

def run_command(cmd, dry_run=False):
    print(f"Running: {cmd}")
    if not dry_run:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
        return result.returncode == 0
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate AACO methods")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    success_count = 0
    total_count = 0

    print("=== Evaluating AACO with Budgets ===")
    for dataset, split in itertools.product(BUDGETS.keys(), SPLITS):
        for budget in BUDGETS[dataset]:
            cmd = f"""uv run scripts/evaluation/eval_afa_method.py \
trained_method_artifact_name=aaco-{dataset}_split_{split}:{ALIAS} \
trained_classifier_artifact_name=null \
+budget={budget} \
dataset_split=validation \
output_artifact_aliases=[\"{ALIAS}\"]"""

            total_count += 1
            if run_command(cmd, args.dry_run):
                success_count += 1

    print(f"Evaluation complete: {success_count}/{total_count} successful")

if __name__ == "__main__":
    main()
