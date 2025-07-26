import itertools
import subprocess

DATASETS = ['cube', 'AFAContext', 'MNIST', 'FashionMNIST', 'diabetes', 'miniboone', 'physionet']
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
    parser = argparse.ArgumentParser(description="Train AACO methods")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    success_count = 0
    total_count = 0

    print("=== Training AACO Methods ===")
    for dataset, split in itertools.product(DATASETS, SPLITS):
        cmd = f"uv run scripts/train_methods/train_aaco.py dataset_artifact_name={dataset}_split_{split}:{ALIAS} output_artifact_aliases=[\"{ALIAS}\"]"
        total_count += 1
        if run_command(cmd, args.dry_run):
            success_count += 1

    print(f"Training complete: {success_count}/{total_count} successful")

if __name__ == "__main__":
    main()
