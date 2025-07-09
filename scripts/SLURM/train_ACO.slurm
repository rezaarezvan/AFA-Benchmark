#!/bin/bash
#SBATCH -A NAISS2025-22-448         # Project account
#SBATCH -p alvis 			        # Queue/partiton name
#SBATCH -N 1 				        # Number of nodes
#SBATCH --gpus-per-node=A40:1		# Number of gpus and types
#SBATCH -t 24:00:00			        # Max runtime
#SBATCH --job-name=aco_test         # Job name
#SBATCH --output=aco_test_%j.log	# Output log file
#SBATCH --error=aco_test_%j.err     # Error log file (debugging)

set -e
echo "Running on $(hostname)"
echo "Start time: $(date)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

module purge
module load virtualenv/20.26.2-GCCcore-13.3.0
module load Python/3.12.3-GCCcore-13.3.0

source .venv/bin/activate

export PYTHONPATH=$PYTHONPATH:$(pwd)

python3 scripts/train_methods/train_aco.py

echo "End time: $(date)"
