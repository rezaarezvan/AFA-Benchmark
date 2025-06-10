#!/usr/bin/env fish

set -g fish_trace 1

# -----------------------
# Parse args
# -----------------------

argparse "dataset=+" "budgets=+" "split=+" "help" "launcher=?" "device=?" "speed=?" "alias=?" -- $argv
or return

# Print help if specified
if set -ql _flag_h
    echo "Usage: README_example.fish --datasets=<list of str> --budgets=<list of str> --splits=<list of int> [-h | --help] [-l | --launcher={custom_slurm,basic}] [-d | --device={cuda,cpu}] [-a | --alias=<alias>" >&2
    return 1
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

set -l alias example
set -ql _flag_alias
and set alias $_flag_alias

set -l speed_suffix
if test "$speed" = "slow"
    set speed_suffix ""
else if test "$speed" = "fast"
    set speed_suffix "_fast"
else
    echo "Third argument should either be 'slow' or 'fast'"
    exit 1
end

set -l extra_opts "device=$device hydra/launcher=$launcher"

# --- DATA GENERATION ---
echo "Starting data generation job..."
sleep 1
uv run scripts/data_generation/generate_dataset.py -m dataset=$(string join , $datasets) split_idx=$(string join , $splits) output_artifact_aliases=["$alias"] hydra/launcher=$launcher

# # --- PRE-TRAINING ---
echo "Starting pretraining jobs..."
sleep 1

set -l jobs
for dataset in $datasets
    set -l dataset_artifact_names
    for split in $splits
        set -a dataset_artifact_names {$dataset}_split_$split:$alias
    end
    set -a jobs "uv run scripts/pretrain_models/pretrain_shim2018.py -m output_artifact_aliases=[\"$alias\"] dataset@_global_=$dataset$speed_suffix dataset_artifact_name=$(string join , $dataset_artifact_names) $extra_opts"
end

mprocs $jobs

# --- METHOD TRAINING ---
echo "Starting method training jobs..."
sleep 1

set -l jobs
set -l i 1
for dataset in $datasets
    set -l dataset_budgets $budgets[$i]
    set -l pretrained_model_artifact_names
    for split in $splits
        set -a pretrained_model_artifact_names pretrain_shim2018-{$dataset}_split_$split:$alias
    end
    set -a jobs "uv run scripts/train_methods/train_shim2018.py -m output_artifact_aliases=[\"$alias\"] dataset@_global_=$dataset$speed_suffix pretrained_model_artifact_name=$(string join , $pretrained_model_artifact_names) hard_budget=$(string join , $dataset_budgets) $extra_opts"
    set i (math "$i+1")
end

mprocs $jobs

# --- CLASSIFIER TRAINING ---
echo "Starting classifier training jobs..."
sleep 1

set -l jobs
for dataset in $datasets
    set -l dataset_artifact_names
    for split in $splits
        set -a dataset_artifact_names {$dataset}_split_$split:$alias
    end
    set -a jobs "uv run scripts/train_classifiers/train_masked_mlp_classifier.py -m output_artifact_aliases=[\"$alias\"] dataset@_global_=$dataset$speed_suffix dataset_artifact_name=$(string join , $dataset_artifact_names) $extra_opts"
end

mprocs $jobs

# --- EVALUATION ---
echo "Starting evaluation jobs..."
sleep 1

set -l i 1
set -l jobs
for dataset in $datasets
    set -l dataset_budgets $budgets[$i]
    for split in $splits
        set -l trained_method_artifact_names
        for budget in (string split , $dataset_budgets)
            set -a trained_method_artifact_names train_shim2018-{$dataset}_split_$split-budget_$budget-seed_42:$alias
        end
        set -a jobs "uv run scripts/evaluation/eval_afa_method.py -m \
          output_artifact_aliases=[\"$alias\"] \
          $(if test $speed = fast; echo eval_only_n_samples=10; end) \
          trained_method_artifact_name=$(string join , $trained_method_artifact_names) \
          trained_classifier_artifact_name=masked_mlp_classifier-$dataset""_split_$split:$alias,null hydra/launcher=$launcher"
        set i (math "$i+1")
    end
end

mprocs $jobs

# --- PLOTTING ---
echo "Starting plotting job..."
sleep 1

set -l eval_artifact_names
set -l i 1
for dataset in $datasets
    set -l dataset_budgets $budgets[$i]
    for split in $splits
        for classifier_artifact_name in "builtin" "masked_mlp_classifier-$dataset""_split_$split"
            for budget in $(string split , $dataset_budgets)
                set -a eval_artifact_names train_shim2018-{$dataset}_split_$split-budget_$budget-seed_42-$classifier_artifact_name:$alias
            end
        end
    end
    set i (math "$i+1")
end
        
uv run scripts/plotting/plot_results.py eval_artifact_names=[$(string join , $eval_artifact_names)]
