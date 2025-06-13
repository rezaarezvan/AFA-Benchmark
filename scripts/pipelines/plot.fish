#!/usr/bin/env fish

# set -g fish_trace 1

set -gx WANDB_ENTITY afa-team
set -gx WANDB_PROJECT afa-benchmark

# -----------------------
# Parse args
# -----------------------

argparse "help" "wandb-entity=?" "wandb-project=?" -- $argv
or exit 1

# Print help if specified
if set -q _flag_help
    echo "Usage: plot.fish [--help] [--wandb-entity=<str>] [--wandb-project=<str>]" >&2
    exit 1
end

# Default arguments

set -l wandb_entity afa-team
set -q _flag_wandb_entity
and set wandb_entity $_flag_wandb_entity

set -l wandb_project afa-benchmark
set -q _flag_wandb_project
and set wandb_project $_flag_wandb_project

set -gx WANDB_ENTITY $wandb_entity
set -gx WANDB_PROJECT $wandb_project

# --- PLOTTING ---
echo "Starting plotting job..."
sleep 1

# Automated approach
# set -l eval_artifact_names
# set -l i 1
# for dataset in $datasets
#     set -l dataset_budgets $budgets[$i]
#     for split in $splits
#         for classifier_artifact_name in "builtin" "masked_mlp_classifier-$dataset""_split_$split"
#             for budget in (string split , $dataset_budgets)
#                 for method in $methods
#                     set -a eval_artifact_names train_$method-{$dataset}_split_$split-budget_$budget-seed_42-$classifier_artifact_name:$evaluation_alias
#                 end
#             end
#         end
#     end
#     set i (math "$i+1")
# end

# Manual approach
set -l eval_artifact_names \
    train_shim2018-cube_split_1-budget_3-seed_42-masked_mlp_classifier-cube_split_1:Jun13-c \
    train_shim2018-cube_split_1-budget_5-seed_42-masked_mlp_classifier-cube_split_1:Jun13-c \
    train_shim2018-cube_split_1-budget_10-seed_42-masked_mlp_classifier-cube_split_1:Jun13-c \
    train_shim2018-cube_split_1-budget_3-seed_42-builtin:Jun13-c \
    train_shim2018-cube_split_1-budget_5-seed_42-builtin:Jun13-c \
    train_shim2018-cube_split_1-budget_10-seed_42-builtin:Jun13-c \
    \
    train_randomdummy-cube_split_1-budget_3-seed_42-masked_mlp_classifier-cube_split_1:Jun13-c \
    train_randomdummy-cube_split_1-budget_5-seed_42-masked_mlp_classifier-cube_split_1:Jun13-c \
    train_randomdummy-cube_split_1-budget_10-seed_42-masked_mlp_classifier-cube_split_1:Jun13-c \
    train_randomdummy-cube_split_1-budget_3-seed_42-builtin:Jun13-c \
    train_randomdummy-cube_split_1-budget_5-seed_42-builtin:Jun13-c \
    train_randomdummy-cube_split_1-budget_10-seed_42-builtin:Jun13-c \
    \
    train_shim2018-cube_split_2-budget_3-seed_42-masked_mlp_classifier-cube_split_2:Jun13-c \
    train_shim2018-cube_split_2-budget_5-seed_42-masked_mlp_classifier-cube_split_2:Jun13-c \
    train_shim2018-cube_split_2-budget_10-seed_42-masked_mlp_classifier-cube_split_2:Jun13-c \
    train_shim2018-cube_split_2-budget_3-seed_42-builtin:Jun13-c \
    train_shim2018-cube_split_2-budget_5-seed_42-builtin:Jun13-c \
    train_shim2018-cube_split_2-budget_10-seed_42-builtin:Jun13-c \
    \
    train_randomdummy-cube_split_2-budget_3-seed_42-masked_mlp_classifier-cube_split_2:Jun13-c \
    train_randomdummy-cube_split_2-budget_5-seed_42-masked_mlp_classifier-cube_split_2:Jun13-c \
    train_randomdummy-cube_split_2-budget_10-seed_42-masked_mlp_classifier-cube_split_2:Jun13-c \
    train_randomdummy-cube_split_2-budget_3-seed_42-builtin:Jun13-c \
    train_randomdummy-cube_split_2-budget_5-seed_42-builtin:Jun13-c \
    train_randomdummy-cube_split_2-budget_10-seed_42-builtin:Jun13-c \
    \
    train_shim2018-MNIST_split_1-budget_10-seed_42-masked_mlp_classifier-MNIST_split_1:Jun13-c \
    train_shim2018-MNIST_split_1-budget_50-seed_42-masked_mlp_classifier-MNIST_split_1:Jun13-c \
    train_shim2018-MNIST_split_1-budget_100-seed_42-masked_mlp_classifier-MNIST_split_1:Jun13-c \
    train_shim2018-MNIST_split_1-budget_10-seed_42-builtin:Jun13-c \
    train_shim2018-MNIST_split_1-budget_50-seed_42-builtin:Jun13-c \
    train_shim2018-MNIST_split_1-budget_100-seed_42-builtin:Jun13-c \
    \
    train_randomdummy-MNIST_split_1-budget_10-seed_42-masked_mlp_classifier-MNIST_split_1:Jun13-c \
    train_randomdummy-MNIST_split_1-budget_50-seed_42-masked_mlp_classifier-MNIST_split_1:Jun13-c \
    train_randomdummy-MNIST_split_1-budget_100-seed_42-masked_mlp_classifier-MNIST_split_1:Jun13-c \
    train_randomdummy-MNIST_split_1-budget_10-seed_42-builtin:Jun13-c \
    train_randomdummy-MNIST_split_1-budget_50-seed_42-builtin:Jun13-c \
    train_randomdummy-MNIST_split_1-budget_100-seed_42-builtin:Jun13-c \
    \
    train_shim2018-MNIST_split_2-budget_10-seed_42-masked_mlp_classifier-MNIST_split_2:Jun13-c \
    train_shim2018-MNIST_split_2-budget_50-seed_42-masked_mlp_classifier-MNIST_split_2:Jun13-c \
    train_shim2018-MNIST_split_2-budget_100-seed_42-masked_mlp_classifier-MNIST_split_2:Jun13-c \
    train_shim2018-MNIST_split_2-budget_10-seed_42-builtin:Jun13-c \
    train_shim2018-MNIST_split_2-budget_50-seed_42-builtin:Jun13-c \
    train_shim2018-MNIST_split_2-budget_100-seed_42-builtin:Jun13-c \
    \
    train_randomdummy-MNIST_split_2-budget_10-seed_42-masked_mlp_classifier-MNIST_split_2:Jun13-c \
    train_randomdummy-MNIST_split_2-budget_50-seed_42-masked_mlp_classifier-MNIST_split_2:Jun13-c \
    train_randomdummy-MNIST_split_2-budget_100-seed_42-masked_mlp_classifier-MNIST_split_2:Jun13-c \
    train_randomdummy-MNIST_split_2-budget_10-seed_42-builtin:Jun13-c \
    train_randomdummy-MNIST_split_2-budget_50-seed_42-builtin:Jun13-c \
    train_randomdummy-MNIST_split_2-budget_100-seed_42-builtin:Jun13-c
        
uv run scripts/plotting/plot_results.py eval_artifact_names=[(string join , $eval_artifact_names)]
