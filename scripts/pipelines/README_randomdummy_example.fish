#!/usr/bin/env fish

# -----------------------
# Parse args
# -----------------------

argparse "dataset=+" "budgets=+" "split=+" "help" "launcher=?" "alias=?" -- $argv
or return

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

set -l splits 1 2
set -ql _flag_split
and set splits $_flag_split

set -l launcher custom_slurm
set -ql _flag_launcher
and set launcher $_flag_launcher

set -l alias example
set -ql _flag_alias
and set alias $_flag_alias

set -l extra_opts "hydra/launcher=$launcher"

# --- METHOD TRAINING ---
echo "Starting method training jobs..."
sleep 1

set -l jobs
set -l i 1
for dataset in $datasets
    set -l dataset_budgets $budgets[$i]
    set -l dataset_artifact_names
    for split in $splits
        set -a dataset_artifact_names {$dataset}_split_$split:$alias
    end
    set -a jobs "uv run scripts/train_methods/train_randomdummy.py -m output_artifact_aliases=[\"$alias\"] dataset_artifact_name=$(string join , $dataset_artifact_names) hard_budget=$(string join , $dataset_budgets) $extra_opts"
    set i (math "$i+1")
end

mprocs $jobs
