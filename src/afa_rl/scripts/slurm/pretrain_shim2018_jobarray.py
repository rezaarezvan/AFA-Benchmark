"""
Runs pretrain_shim2018 on all combinations of dataset_type, split and seed.
"""


import subprocess
from coolname import generate_slug

DATASET_TYPES = ["cube", "diabetes"]
DATASET_SPLITS = [1, 2, 3]
SEEDS = [42, 43, 44]

for dataset_type in DATASET_TYPES:
    for split in DATASET_SPLITS:
        for seed in SEEDS:
            subprocess.run(
                [
                    "sbatch",
                    "--export=ALL",
                    "split=1",
                    f"dataset_type={dataset_type}",
                    f"train_dataset_path=data/{dataset_type}/train_split_{split}.pt",
                    f"val_dataset_path=data/{dataset_type}/val_split_{split}.pt",
                    f"pretrained_model_path=models/pretrained/shim2018/{generate_slug(2)}",
                    f"seed={seed}",
                    "src/afa_rl/scripts/slurm/pretrain_shim2018_job.sh",
                ],
                shell=True,
            )
