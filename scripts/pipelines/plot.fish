#!/usr/bin/env fish

# set -g fish_trace 1

set -gx WANDB_ENTITY afa-team
set -gx WANDB_PROJECT afa-benchmark

# -----------------------
# Parse args
# -----------------------

argparse "help" "output-alias=?" "wandb-entity=?" "wandb-project=?" -- $argv
or exit 1

# Print help if specified
if set -ql _flag_help
    echo "Usage: plot.fish [--help] [--output-alias=<str>] [--wandb-entity=<str>] [--wandb-project=<str>]" >&2
    exit 1
end

# Default arguments

set -l output_alias tmp
set -ql _flag_output_alias
and set output_alias $_flag_output_alias

set -l wandb_entity afa-team
set -ql _flag_wandb_entity
and set wandb_entity $_flag_wandb_entity

set -l wandb_project afa-benchmark
set -ql _flag_wandb_project
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
    train_shim2018_cube_split_1-budget_3_seed_42-masked_mlp_classifier-cube_split_1:Jun12 \
    train_shim2018_cube_split_1-budget_5_seed_42-masked_mlp_classifier-cube_split_1:Jun12 \
    train_shim2018_cube_split_1-budget_10_seed_42-masked_mlp_classifier-cube_split_1:Jun12 \
    train_shim2018_cube_split_1-budget_3_seed_42-builtin:Jun12 \
    train_shim2018_cube_split_1-budget_5_seed_42-builtin:Jun12 \
    train_shim2018_cube_split_1-budget_10_seed_42-builtin:Jun12 \
    \
    train_randomdummy_cube_split_1-budget_3_seed_42-masked_mlp_classifier-cube_split_1:Jun12 \
    train_randomdummy_cube_split_1-budget_5_seed_42-masked_mlp_classifier-cube_split_1:Jun12 \
    train_randomdummy_cube_split_1-budget_10_seed_42-masked_mlp_classifier-cube_split_1:Jun12 \
    train_randomdummy_cube_split_1-budget_3_seed_42-builtin:Jun12 \
    train_randomdummy_cube_split_1-budget_5_seed_42-builtin:Jun12 \
    train_randomdummy_cube_split_1-budget_10_seed_42-builtin:Jun12 \
    \
    train_shim2018_cube_split_2-budget_3_seed_42-masked_mlp_classifier-cube_split_2:Jun12 \
    train_shim2018_cube_split_2-budget_5_seed_42-masked_mlp_classifier-cube_split_2:Jun12 \
    train_shim2018_cube_split_2-budget_10_seed_42-masked_mlp_classifier-cube_split_2:Jun12 \
    train_shim2018_cube_split_2-budget_3_seed_42-builtin:Jun12 \
    train_shim2018_cube_split_2-budget_5_seed_42-builtin:Jun12 \
    train_shim2018_cube_split_2-budget_10_seed_42-builtin:Jun12 \
    \
    train_randomdummy_cube_split_2-budget_3_seed_42-masked_mlp_classifier-cube_split_2:Jun12 \
    train_randomdummy_cube_split_2-budget_5_seed_42-masked_mlp_classifier-cube_split_2:Jun12 \
    train_randomdummy_cube_split_2-budget_10_seed_42-masked_mlp_classifier-cube_split_2:Jun12 \
    train_randomdummy_cube_split_2-budget_3_seed_42-builtin:Jun12 \
    train_randomdummy_cube_split_2-budget_5_seed_42-builtin:Jun12 \
    train_randomdummy_cube_split_2-budget_10_seed_42-builtin:Jun12 \
    \
    train_shim2018_MNIST_split_1-budget_10_seed_42-masked_mlp_classifier-MNIST_split_1:Jun12 \
    train_shim2018_MNIST_split_1-budget_50_seed_42-masked_mlp_classifier-MNIST_split_1:Jun12 \
    train_shim2018_MNIST_split_1-budget_100_seed_42-masked_mlp_classifier-MNIST_split_1:Jun12 \
    train_shim2018_MNIST_split_1-budget_10_seed_42-builtin:Jun12 \
    train_shim2018_MNIST_split_1-budget_50_seed_42-builtin:Jun12 \
    train_shim2018_MNIST_split_1-budget_100_seed_42-builtin:Jun12 \
    \
    train_randomdummy_MNIST_split_1-budget_10_seed_42-masked_mlp_classifier-MNIST_split_1:Jun12 \
    train_randomdummy_MNIST_split_1-budget_50_seed_42-masked_mlp_classifier-MNIST_split_1:Jun12 \
    train_randomdummy_MNIST_split_1-budget_100_seed_42-masked_mlp_classifier-MNIST_split_1:Jun12 \
    train_randomdummy_MNIST_split_1-budget_10_seed_42-builtin:Jun12 \
    train_randomdummy_MNIST_split_1-budget_50_seed_42-builtin:Jun12 \
    train_randomdummy_MNIST_split_1-budget_100_seed_42-builtin:Jun12 \
    \
    train_shim2018_MNIST_split_2-budget_10_seed_42-masked_mlp_classifier-MNIST_split_2:Jun12 \
    train_shim2018_MNIST_split_2-budget_50_seed_42-masked_mlp_classifier-MNIST_split_2:Jun12 \
    train_shim2018_MNIST_split_2-budget_100_seed_42-masked_mlp_classifier-MNIST_split_2:Jun12 \
    train_shim2018_MNIST_split_2-budget_10_seed_42-builtin:Jun12 \
    train_shim2018_MNIST_split_2-budget_50_seed_42-builtin:Jun12 \
    train_shim2018_MNIST_split_2-budget_100_seed_42-builtin:Jun12 \
    \
    train_randomdummy_MNIST_split_2-budget_10_seed_42-masked_mlp_classifier-MNIST_split_2:Jun12 \
    train_randomdummy_MNIST_split_2-budget_50_seed_42-masked_mlp_classifier-MNIST_split_2:Jun12 \
    train_randomdummy_MNIST_split_2-budget_100_seed_42-masked_mlp_classifier-MNIST_split_2:Jun12 \
    train_randomdummy_MNIST_split_2-budget_10_seed_42-builtin:Jun12 \
    train_randomdummy_MNIST_split_2-budget_50_seed_42-builtin:Jun12 \
    train_randomdummy_MNIST_split_2-budget_100_seed_42-builtin:Jun12
        
uv run scripts/plotting/plot_results.py eval_artifact_names=[(string join , $eval_artifact_names)]
