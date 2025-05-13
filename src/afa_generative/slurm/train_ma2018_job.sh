#!/usr/bin/bash
#SBATCH --account=NAISS2025-22-448 -p alvis
#SBATCH -N 1 --gpus-per-node=T4:1
#SBATCH -t 24:00:00
#SBATCH --output=/mimer/NOBACKUP/groups/active-learning/projects/AFA-Benchmark/logs/slurm/train_ma2018_%j.out

module load virtualenv

uv run src/afa_generative/scripts/train_ma2018.py \
    --pretrain_config configs/ma2018/pretrain_ma2018.yml \
    --train_config configs/ma2018/train_ma2018.yml \
    --dataset_type $dataset_type \
    --train_dataset_path $train_dataset_path \
    --val_dataset_path $val_dataset_path \
    --pretrained_model_path $pretrained_model_path \
    --hard_budget $hard_budget \
    --seed $seed \
    --afa_method_path $afa_method_path && \
echo "Saved model to $afa_method_path. Writing to $status_file ..."
echo "success" > $status_file
