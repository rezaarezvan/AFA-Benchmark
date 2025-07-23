#!/bin/bash
#SBATCH -A NAISS2025-22-448                 # Project account
#SBATCH -p alvis 			                # Queue/partiton name
#SBATCH -N 1 				                # Number of nodes
#SBATCH --gpus-per-node=T4:1		        # Number of gpus and types
#SBATCH -t 24:00:00			                # Max runtime
#SBATCH --job-name=aaco_train               # Job name
#SBATCH --output=logs/aaco_train_%j.out     # Output log file
#SBATCH --error=logs/aaco_train_%j.err      # Error log file (debugging)

module purge
module load virtualenv/20.26.2-GCCcore-13.3.0
module load Python/3.12.3-GCCcore-13.3.0

# Default values
DATASET=${1:-"MNIST"}
DEVICE=${3:-"cuda"}

set -e
echo "Running on $(hostname)"
echo "Start time: $(date)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo "Training AACO with dataset: $DATASET, seed: $SEED, device: $DEVICE"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODEID"

source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run training
uv run scripts/train_methods/train_aco.py \
    +dataset=$DATASET \
    device=$DEVICE \

echo "Training completed for $DATASET"
echo "End time: $(date)"
