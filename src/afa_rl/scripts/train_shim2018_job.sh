#!/usr/bin/bash
#SBATCH --account=NAISS2025-22-448 -p alvis
#SBATCH -N 1 --gpus-per-node=T4:1
#SBATCH -t 24:00:00

split=1

echo "Training agent for split $split..."
uv run src/afa_rl/scripts/train_shim2018.py \
    --pretrain_config configs/shim2018/pretrain_shim2018.yml \
    --train_config configs/shim2018/train_shim2018.yml \
    --dataset_type "cube" \
    --dataset_train_path "data/cube/train_split_$split.pt" \
    --dataset_val_path "data/cube/val_split_$split.pt" \
    --pretrained_model_path "models/shim2018/pretrained/shim2018-cube_train_split_$split.pt" \
    --afa_method_path "models/shim2018/shim2018-cube_train_split_$split.pt"
echo "Finished training agent for split $split."
