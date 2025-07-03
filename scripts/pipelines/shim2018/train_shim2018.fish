#!/usr/bin/env fish

# set -g fish_trace 1

# -----------------------
# Parse args
# -----------------------

argparse "dataset=+" "budgets=+" "split=+" "help" "launcher=?" "device=?" "speed=?" "pretrain-alias=?" "output-alias=?" "wandb-entity=?" "wandb-project=?" -- $argv
or exit 1

# Print help if specified
if set -ql _flag_help
    echo "Usage: train_shim2018.fish --dataset=<str> --budgets=<str> --split=<int> [--help] [--launcher={custom_slurm,basic}] [--device={cuda,cpu}] [--pretrain-alias=<str>] [--output-alias=<str>] [--wandb-entity=<str>] [--wandb-project=<str>]" >&2
    exit 1
end

# Default arguments

# No default arguments for datasets
if set -q _flag_dataset
    set datasets $_flag_dataset
else
    echo "datasets must be set"
    exit 1
end

# No default argument for budgets
if set -q _flag_budgets
    set budgets $_flag_budgets
else
    echo "budgets must be set"
    exit 1
end

set -l splits 1 2
set -q _flag_split
and set splits $_flag_split

set -l launcher custom_slurm
set -q _flag_launcher
and set launcher $_flag_launcher

set -l device cuda
set -q _flag_device
and set device $_flag_device

set -l speed slow
set -q _flag_speed
and set speed $_flag_speed

set -l pretrain_alias tmp
set -q _flag_pretrain_alias
and set pretrain_alias $_flag_pretrain_alias

set -l output_alias tmp
set -q _flag_output_alias
and set output_alias $_flag_output_alias

set -l speed_suffix
if test "$speed" = "slow"
    set speed_suffix ""
else if test "$speed" = ""
    echo "Speed argument should either be 'slow', 'medium' or 'fast'"
    exit 1
else
    set speed_suffix "_$speed"
end

set -l wandb_entity afa-team
set -q _flag_wandb_entity
and set wandb_entity $_flag_wandb_entity

set -l wandb_project afa-benchmark
set -q _flag_wandb_project
and set wandb_project $_flag_wandb_project

set -gx WANDB_ENTITY $wandb_entity
set -gx WANDB_PROJECT $wandb_project

set -l extra_opts "device=$device hydra/launcher=$launcher"

# --- METHOD TRAINING ---
echo "Starting shim2018 method training jobs..."
sleep 1

set -l jobs
set -l i 1
for dataset in $datasets
    set -l dataset_budgets $budgets[$i]
    set -l pretrained_model_artifact_names
    for split in $splits
        set -a pretrained_model_artifact_names pretrain_shim2018-{$dataset}_split_$split:$pretrain_alias
    end
    set -a jobs "uv run scripts/train_methods/train_shim2018.py -m output_artifact_aliases=[\"$output_alias\"] dataset@_global_=$dataset$speed_suffix pretrained_model_artifact_name="(string join , $pretrained_model_artifact_names)" hard_budget="(string join , $dataset_budgets)" $extra_opts"
    set i (math "$i+1")
end

mprocs $jobs
