import itertools
import subprocess

DATASETS = ['cube', 'AFAContext', 'MNIST',
            'FashionMNIST', 'diabetes', 'miniboone', 'physionet']
SPLITS = [1, 2, 3, 4, 5]
INPUT_ALIAS = "latest"
OUTPUT_ALIAS = "KDD"


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Launch AACO training jobs on SLURM")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running")
    args = parser.parse_args()

    print("=== Launching AACO Training Jobs ===")

    for dataset, split in itertools.product(DATASETS, SPLITS):
        cmd = f"""uv run scripts/train_methods/train_aaco.py -m \
dataset_artifact_name={dataset}_split_{split}:{INPUT_ALIAS} \
output_artifact_aliases=["{OUTPUT_ALIAS}"] \
hydra/launcher=custom_slurm"""

        print(f"Submitting: {dataset}_split_{split}")
        if not args.dry_run:
            subprocess.Popen(cmd, shell=True)  # Non-blocking

    print("All training jobs submitted to SLURM")


if __name__ == "__main__":
    main()
