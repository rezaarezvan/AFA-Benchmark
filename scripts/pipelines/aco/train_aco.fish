#!/usr/bin/env fish
# ACO Training Pipeline Script
# Usage: fish scripts/pipelines/aco/train_aco.fish --dataset=cube --wandb-entity=your_entity --wandb-project=your_project

argparse 'dataset=' 'wandb-entity=' 'wandb-project=' 'alias=' 'device=' 'seed=' -- $argv
or exit 1

# Set defaults
set -q _flag_dataset; or set _flag_dataset "cube"
set -q _flag_device; or set _flag_device "cuda"
set -q _flag_seed; or set _flag_seed "42"
set -q _flag_alias; or set _flag_alias "test"

# Validate required arguments
if not set -q _flag_wandb_entity
    echo "Error: --wandb-entity is required"
    exit 1
end

if not set -q _flag_wandb_project
    echo "Error: --wandb-project is required"
    exit 1
end

echo "Training ACO method with:"
echo "  Dataset: $_flag_dataset"
echo "  Device: $_flag_device"
echo "  Seed: $_flag_seed"
echo "  Alias: $_flag_alias"
echo "  Wandb Entity: $_flag_wandb_entity"
echo "  Wandb Project: $_flag_wandb_project"

# Set environment variables
set -x WANDB_ENTITY $_flag_wandb_entity
set -x WANDB_PROJECT $_flag_wandb_project

# Run training
echo "Starting ACO training..."
uv run scripts/train_methods/train_aco.py \
    dataset=$_flag_dataset \
    seed=$_flag_seed \
    device=$_flag_device \
    output_artifact_aliases=[$_flag_alias] \
    dataset_artifact_name="${_flag_dataset}_split_1:$_flag_alias"

echo "ACO training completed!"

# Optional: Run evaluation
set -q _flag_eval; and begin
    echo "Running evaluation..."
    uv run scripts/evaluation/eval_afa_method.py \
        method_artifact_name="train_aco-$_flag_dataset-seed_$_flag_seed:$_flag_alias" \
        output_artifact_aliases=[$_flag_alias]
end
