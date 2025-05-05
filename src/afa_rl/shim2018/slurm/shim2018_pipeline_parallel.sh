#!/usr/bin/bash
#SBATCH --account=NAISS2025-22-448 -p alvis
#SBATCH -N 1 --gpus-per-node=T4:1
#SBATCH -t 24:00:00
#SBATCH --output=/mimer/NOBACKUP/groups/meta-project/projects/AFA-Benchmark/logs/slurm/shim2018_pipeline_sequential_%j.out

module load virtualenv

uv run src/afa_rl/scripts/shim2018_pipeline_sequential.py
