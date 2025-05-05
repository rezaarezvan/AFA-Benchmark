#!/usr/bin/bash
#SBATCH --account=NAISS2025-22-448 -p alvis
#SBATCH -N 1 --gpus-per-node=T4:1
#SBATCH --time 24:00:00
#SBATCH --output=/mimer/NOBACKUP/groups/meta-project/projects/AFA-Benchmark/logs/slurm/pretrain_shim2018_%j.out

module load virtualenv

uv run src/afa_rl/shim2018/scripts/pretrain_shim2018.py \
    --pretrain_config configs/shim2018/pretrain_shim2018.yml \
    --dataset_type $dataset_type \
    --train_dataset_path $train_dataset_path \
    --val_dataset_path $val_dataset_path \
    --pretrained_model_path $pretrained_model_path \
    --seed $seed
