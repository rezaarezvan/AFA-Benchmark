#!/usr/bin/env fish

# set -g fish_trace 1

# -----------------------
# Parse args
# -----------------------

argparse "dataset=+" "split=+" "help" "launcher=?" "device=?" "speed=?" "dataset-alias=?" "output-alias=?" "wandb-entity=?" "wandb-project=?" -- $argv
or exit 1

# Print help if specified
if set -ql _flag_help
    echo "Usage: train_classifier.fish --dataset=<str> --split=<int> [--help] [--launcher={custom_slurm,basic}] [--device={cuda,cpu}] [--dataset-alias=<str>] [--output-alias=<str>] [--wandb-entity=<str>] [--wandb-project=<str>]" >&2
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

set -l splits 1 2
set -ql _flag_split
and set splits $_flag_split

set -l launcher custom_slurm
set -ql _flag_launcher
and set launcher $_flag_launcher

set -l device cuda
set -ql _flag_device
and set device $_flag_device

set -l speed slow
set -ql _flag_speed
and set speed $_flag_speed

set -l dataset_alias tmp
set -ql _flag_dataset_alias
and set dataset_alias $_flag_dataset_alias

set -l output_alias tmp
set -ql _flag_output_alias
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
set -ql _flag_wandb_entity
and set wandb_entity $_flag_wandb_entity

set -l wandb_project afa-benchmark
set -ql _flag_wandb_project
and set wandb_project $_flag_wandb_project

set -gx WANDB_ENTITY $wandb_entity
set -gx WANDB_PROJECT $wandb_project

set -l extra_opts "device=$device hydra/launcher=$launcher"

# --- CLASSIFIER TRAINING ---
echo "Starting classifier training jobs..."
sleep 1

set -l jobs
for dataset in $datasets
    set -l dataset_artifact_names
    for split in $splits
        set -a dataset_artifact_names {$dataset}_split_$split:$dataset_alias
    end
    set -a jobs "uv run scripts/train_classifiers/train_masked_mlp_classifier.py -m output_artifact_aliases=[\"$output_alias\"] dataset@_global_=$dataset$speed_suffix dataset_artifact_name="(string join , $dataset_artifact_names)" $extra_opts"
end

mprocs $jobs
