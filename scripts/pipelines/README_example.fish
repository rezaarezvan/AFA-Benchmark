#!/usr/bin/env fish

set -g fish_trace 1

# -----------------------
# Parse args
# -----------------------

argparse "help" "launcher=?" "device=?" "speed=?" "wandb-entity=?" "wandb-project=?" "dataset-alias=?" "alias=?" -- $argv
or exit 1

# Print help if specified
if set -ql _flag_help
    echo "Usage: README_example.fish [--help] [--launcher={custom_slurm,basic}] [--device={cuda,cpu}] [--speed=<str>] [--wandb-entity=<str>] [--wandb-project=<str>] [--dataset-alias=<str>] [--alias=<str>]" >&2
    exit 1
end

# Default arguments

set -l launcher custom_slurm
set -ql _flag_launcher
and set launcher $_flag_launcher

set -l device cuda
set -ql _flag_device
and set device $_flag_device

set -l speed slow
set -ql _flag_speed
and set speed $_flag_speed

set -l speed_suffix
if test "$speed" = "slow"
    set speed_suffix ""
elif test "$speed" = ""
    echo "Speed argument should either be 'slow', 'medium' or 'fast'"
    exit 1
else
    set speed_suffix "_$speed"
end

set -ql _flag_wandb_entity
and set wandb_entity $_flag_wandb_entity

set -l wandb_project afa-benchmark
set -ql _flag_wandb_project
and set wandb_project $_flag_wandb_project

set -gx WANDB_ENTITY $wandb_entity
set -gx WANDB_PROJECT $wandb_project

set -l dataset_alias tmp
set -ql _flag_dataset_alias
and set dataset_alias $_flag_dataset_alias

set -l alias example
set -ql _flag_alias
and set alias $_flag_alias

# Hard-coded variables for this example
set -l datasets cube MNIST
set -l budgets 3,5,10 10,50,100

# Assume data generation is already done

# # --- PRE-TRAINING ---
./scripts/pipelines/shim2018/pretrain_shim2018.fish --dataset=$datasets[1] --dataset=$datasets[2] --split=1 --split=2 --launcher=$launcher --device=$device --speed=$speed --dataset-alias=$dataset_alias --output-alias=$alias --wandb-entity=$wandb_entity --wandb-project=$wandb_project


# --- METHOD TRAINING ---
# shim2018
./scripts/pipelines/shim2018/train_shim2018.fish --dataset=$datasets[1] --dataset=$datasets[2] --budgets=$budgets[1] --budgets=$budgets[2] --split=1 --split=2 --launcher=$launcher --device=$device --speed=$speed --pretrain-alias=$alias --output-alias=$alias --wandb-entity=$wandb_entity --wandb-project=$wandb_project
# randomdummy
./scripts/pipelines/randomdummy/train_randomdummy.fish --dataset=$datasets[1] --dataset=$datasets[2] --budgets=$budgets[1] --budgets=$budgets[2] --split=1 --split=2 --launcher=$launcher --dataset-alias=$alias --output-alias=$alias --wandb-entity=$wandb_entity --wandb-project=$wandb_project

# --- CLASSIFIER TRAINING ---
./scripts/pipelines/train_classifier.fish --dataset=$datasets[1] --dataset=$datasets[2] --split=1 --split=2 --launcher=$launcher --device=$device --speed=$speed --dataset-alias=$alias --output-alias=$alias --wandb-entity=$wandb_entity --wandb-project=$wandb_project

# --- EVALUATION ---

echo "Starting evaluation jobs..."
sleep 1

set -l i 1
set -l jobs
for dataset in $datasets
    set -l dataset_budgets $budgets[$i]
    for split in 1 2
        set -l trained_method_artifact_names
        for budget in (string split , $dataset_budgets)
            for method in shim2018 randomdummy
                set -a trained_method_artifact_names train_$method-{$dataset}_split_$split-budget_$budget-seed_42:$alias
            end
        end
        set -a jobs "uv run scripts/evaluation/eval_afa_method.py -m \
          output_artifact_aliases=[\"$alias\"] \
          "(if test $speed = slow; echo eval_only_n_samples=10; end)" \
          trained_method_artifact_name="(string join , $trained_method_artifact_names)" \
          trained_classifier_artifact_name=masked_mlp_classifier-$dataset""_split_$split:$alias,null \
          hydra/launcher=$launcher"
    end
    set i (math "$i+1")
end

mprocs $jobs

# --- PLOTTING ---
./scripts/pipelines/plot.fish --dataset=$datasets[1] --dataset=$datasets[2] --budgets=$budgets[1] --budgets=$budgets[2] --method=shim2018 --method=randomdummy --split=1 --split=2 --evaluation-alias=$alias --wandb-entity=$wandb_entity --wandb-project=$wandb_project
