#!/usr/bin/env fish

# set -g fish_trace 1

set -gx WANDB_ENTITY afa-team
set -gx WANDB_PROJECT afa-benchmark

# -----------------------
# Parse args
# -----------------------

argparse "dataset=+" "budgets=+" "method=+" "split=+" "help" "evaluation-alias=?" "output-alias=?" "wandb-entity=?" "wandb-project=?" -- $argv
or exit 1

# Print help if specified
if set -ql _flag_help
    echo "Usage: plot.fish --dataset=<str> --budgets=<str> --method=<str> --split=<int> [--help] [--evaluation-alias=<str>] [--output-alias=<str>] [--wandb-entity=<str>] [--wandb-project=<str>]" >&2
    exit 1
end

# Default arguments

# No default arguments for datasets
if set -ql _flag_dataset
    set datasets $_flag_dataset
else
    echo "datasets must be set"
    exit 1
end

# No default argument for budgets
if set -ql _flag_budgets
    set budgets $_flag_budgets
else
    echo "budgets must be set"
    exit 1
end

set -l methods shim2018 randomdummy
set -ql _flag_method
and set methods $_flag_method

set -l splits 1 2
set -ql _flag_split
and set splits $_flag_split

set -l evaluation_alias tmp
set -ql _flag_evaluation_alias
and set evaluation_alias $_flag_evaluation_alias

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

set -l eval_artifact_names
set -l i 1
for dataset in $datasets
    set -l dataset_budgets $budgets[$i]
    for split in $splits
        for classifier_artifact_name in "builtin" "masked_mlp_classifier-$dataset""_split_$split"
            for budget in (string split , $dataset_budgets)
                for method in $methods
                    set -a eval_artifact_names train_$method-{$dataset}_split_$split-budget_$budget-seed_42-$classifier_artifact_name:$evaluation_alias
                end
            end
        end
    end
    set i (math "$i+1")
end
        
uv run scripts/plotting/plot_results.py eval_artifact_names=[(string join , $eval_artifact_names)]
