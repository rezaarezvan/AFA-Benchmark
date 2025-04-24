#!/usr/bin/bash
#SBATCH --account=NAISS2025-22-448 -p alvis
#SBATCH -N 1 --gpus-per-node=T4:1
#SBATCH -t 24:00:00
#SBATCH --array=1-5

split=$SLURM_ARRAY_TASK_ID

# Ensure the split variable is set correctly
if [ -z "$split" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID is not set. Please run this script as part of a job array."
    exit 1
fi

echo "Pretraining model for split $split..."
uv run src/afa_rl/scripts/pretrain_zannone2019.py \
    --pretrain_config configs/zannone2019/pretrain_zannone2019.yml \
    --dataset_type "cube" \
    --dataset_train_path "data/cube/train_split_$split.pt" \
    --dataset_val_path "data/cube/val_split_$split.pt" \
    --pretrained_model_path "models/zannone2019/pretrained/zannone2019-cube_train_split_$split.pt"
echo "Finished pretraining model for split $split."
