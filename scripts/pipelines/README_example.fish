#!/usr/bin/env fish

set -g fish_trace 1

# -----------------------
# Parse args
# -----------------------

argparse "datasets=" "budgets=" "splits=?" "help" "launcher=?" "device=?" "speed=?" "alias=?" -- $argv
or return

# Print help if specified
if set -ql _flag_h
    echo "Usage: README_example.fish --datasets=<list of str> --budgets=<list of str> --splits=<list of int> [-h | --help] [-l | --launcher={custom_slurm,basic}] [-d | --device={cuda,cpu}] [-a | --alias=<alias>" >&2
    return 1
end

# Default arguments

# No default arguments for datasets
if set -ql _flag_datasets[1]
    set datasets $_flag_datasets
else
    echo "datasets must be set"
    exit 1
end

# No default argument for budgets
if set -ql _flag_budgets[1]
    set budgets $_flag_budgets
else
    echo "budgets must be set"
    exit 1
end

set -l splits 1 2
set -ql _flag_splits[1]
and set splits $_flag_splits[-1]

set -l launcher custom_slurm
set -ql _flag_launcher[1]
and set launcher $_flag_launcher[-1]

set -l device cuda
set -ql _flag_device[1]
and set device $_flag_device[-1]

set -l speed slow
set -ql _flag_speed[1]
and set speed $_flag_speed[-1]

set -l alias example
set -ql _flag_alias[1]
and set alias $_flag_alias[-1]

if test "$speed" = "slow"
    set -l speed_suffix ""
else if test "$speed" = "fast"
    set -l speed_suffix "_fast"
else
    echo "Third argument should either be 'slow' or 'fast'"
    exit 1
end

set -l extra_opts "device=$device hydra/launcher=$launcher"

# -----------------------
# Enable steps based on range
# -----------------------
#TODO
# -----------------------
# Pipeline
# -----------------------


# --- DATA GENERATION ---
echo "Starting data generation job..."
sleep 1
uv run scripts/data_generation/generate_dataset.py -m dataset=$(string join , $datasets) split_idx=$(string join , $splits) output_artifact_aliases=["$alias"] hydra/launcher=$launcher

# --- PRE-TRAINING ---
echo "Starting pretraining jobs..."
sleep 1

set -l jobs
for dataset in $datasets
    set -l dataset_artifact_names
    for split in $splits
        set -la dataset_artifact_names $dataset_split_$split:$alias
    end
    set -la jobs "uv run scripts/pretrain_models/pretrain_shim2018.py -m output_artifact_aliases=[\"$alias\"] dataset@_global_=$dataset$speed_suffix dataset_artifact_name=$(string join , dataset_artifact_names) $extra_opts"
end

mprocs $jobs

# --- METHOD TRAINING ---
echo "Starting method training jobs..."
sleep 1

set -l jobs
set -l i 1
for dataset in $datasets
    set -l dataset_budgets budgets[$i]
    set -l pretrained_model_artifact_names
    for split in $splits
        set -la pretrained_model_artifact_names pretrained_shim2018-$dataset_split_$split:$alias
    end
    set -la jobs "uv run scripts/train_methods/train_shim2018.py -m output_artifact_aliases=[\"$alias\"] dataset@_global_=$dataset$speed_suffix pretrained_model_artifact_name=$(string join , $pretrained_model_artifact_names) hard_budget=$(string join , $dataset_budgets) $extra_opts"
    set -l i (math "$i+1")
end

mprocs $jobs

# --- CLASSIFIER TRAINING ---
echo "Starting classifier training jobs..."
sleep 1

set -l jobs
for dataset in $datasets
    set -l dataset_artifact_names
    for split in $splits
        set -la dataset_artifact_names $dataset_split_$split:$alias
    end
    set -la jobs "uv run scripts/train_classifiers/train_masked_mlp_classifier.py -m output_artifact_aliases=[\"$alias\"] dataset@_global_=$dataset$speed_suffix dataset_artifact_name=$(string join , $dataset_artifact_names) $extra_opts"
end

mprocs $jobs

# --- EVALUATION ---
echo "Starting evaluation jobs..."
sleep 1

set -l i 1
set -l jobs
for dataset in $datasets
    set -l dataset_budgets budgets[$i]
    for split in $splits
        set -l trained_method_artifact_names
        for budget in (string split , $dataset_budgets)
            set -la trained_method_artifact_names train_shim2018-$dataset_split_$split-budget_$budget-seed_42:$alias
        end
        set -la jobs "uv run scripts/evaluation/eval_afa_method.py -m \
          output_artifact_aliases=[\"$alias\"] \
          eval_only_n_samples=100 \
          trained_method_artifact_name=$(string join , $trained_method_artifact_names) \
          trained_classifier_artifact_name=masked_mlp_classifier-$dataset_split_$split:$alias,null hydra/launcher=$launcher"
        set -l i math "$i+1"
    end
end

mprocs $jobs

# --- PLOTTING ---
echo "Starting plotting job..."
sleep 1

set -l eval_artifact_names
set -l i 1
for dataset in $datasets
    set -l dataset_budgets budgets[$i]
    for split in $splits
        for classifier_artifact_name in "builtin:$alias" "masked_mlp_classifier-$dataset_split_$split:$alias"
            for budget in $(string split , dataset_budgets)
                set -la eval_artifact_names train_shim2018-$dataset_split_$split-budget_$budget-seed_42-$classifier_artifact_name
            end
        end
    end
    set -l i math "$i+1"
end
        
uv run scripts/plotting/plot_results.py eval_artifact_names=[$(string join , $eval_artifact_names)]
