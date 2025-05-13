#!/usr/bin/bash
#SBATCH --account=NAISS2025-22-448 -p alvis
#SBATCH -N 1 --gpus-per-node=T4:1
#SBATCH --time 24:00:00
#SBATCH --output=/mimer/NOBACKUP/groups/meta-project/projects/AFA-Benchmark/logs/slurm/pretrain_shim2018_%j.out

module load virtualenv

if uv run src/afa_rl/shim2018/scripts/pretrain_shim2018.py \
    --pretrain_config configs/shim2018/pretrain_shim2018.yml \
    --dataset_type $dataset_type \
    --train_dataset_path $train_dataset_path \
    --val_dataset_path $val_dataset_path \
    --pretrained_model_path $pretrained_model_path \
    --seed $seed; then
    echo "Saved model to $pretrained_model_path. Writing to $status_file ..."
    echo "success" > $status_file

    # Add a suffix to the log file
    suffix="_completed"
    log_file="/mimer/NOBACKUP/groups/meta-project/projects/AFA-Benchmark/logs/slurm/pretrain_shim2018_${SLURM_JOB_ID}.out"
    mv "$log_file" "${log_file}${suffix}"
else
    echo "uv run failed. Writing failure to $status_file ..."
    echo "failure" > $status_file
fi
