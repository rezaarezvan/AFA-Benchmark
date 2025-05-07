#!/usr/bin/bash
#SBATCH --account=NAISS2025-22-448 -p alvis
#SBATCH -N 1 --gpus-per-node=T4:1
#SBATCH -t 24:00:00
#SBATCH --output=/mimer/NOBACKUP/groups/meta-project/projects/AFA-Benchmark/logs/slurm/train_zannone2019_%j.out

module load virtualenv

if uv run src/afa_rl/zannone2019/scripts/train_zannone2019.py \
    --pretrain_config configs/zannone2019/pretrain_zannone2019.yml \
    --train_config configs/zannone2019/train_zannone2019.yml \
    --dataset_type $dataset_type \
    --train_dataset_path $train_dataset_path \
    --val_dataset_path $val_dataset_path \
    --pretrained_model_path $pretrained_model_path \
    --hard_budget $hard_budget \
    --seed $seed \
    --afa_method_path $afa_method_path; then
    echo "Saved model to $afa_method_path. Writing to $status_file ..." && \
    echo "success" > $status_file

    # Add a suffix to the log file
    suffix="_completed"
    log_file="/mimer/NOBACKUP/groups/meta-project/projects/AFA-Benchmark/logs/slurm/train_zannone2019_${SLURM_JOB_ID}.out"
    mv "$log_file" "${log_file}${suffix}"
else
    echo "uv run failed. Writing failure to $status_file ..." && \
    echo "failure" > $status_file
fi
