#!/usr/bin/env fish

set -g fish_trace 1


# -----------------------
# Parse args
# -----------------------

argparse "dataset=+" "split=+" "help" "launcher=?" "output-alias=?" "wandb-entity=?" "wandb-project=?" -- $argv
or exit 1

# Print help if specified
if set -ql _flag_h
    echo "Usage: generate_data.fish --dataset=<str> --split=<int> [--help] [--launcher={custom_slurm,basic}] [--output-alias=<str>] [--wandb-entity=<str>] [--wandb-project=<str>]" >&2
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

set -l extra_opts "hydra/launcher=$launcher"

# --- DATA GENERATION ---
echo "Starting data generation job..."
sleep 1
uv run scripts/data_generation/generate_dataset.py -m dataset=(string join , $datasets) split_idx=(string join , $splits) output_artifact_aliases=["$output_alias"] hydra/launcher=$launcher
