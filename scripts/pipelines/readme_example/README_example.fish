#!/usr/bin/env fish

set -g fish_trace 1

# -----------------------
# Parse args
# -----------------------

argparse "help" "launcher=?" "device=?" "wandb-entity=?" "wandb-project=?" -- $argv
or exit 1

# Print help if specified
if set -ql _flag_h
    echo "Usage: README_example.fish [--help] [--launcher={custom_slurm,basic}] [--device={cuda,cpu}] [--wandb-entity=<str>] [--wandb-project=<str>]" >&2
    exit 1
end

# Default arguments

set -l launcher custom_slurm
set -ql _flag_launcher
and set launcher $_flag_launcher

set -l device cuda
set -ql _flag_device
and set device $_flag_device

set -ql _flag_wandb_entity
and set wandb_entity $_flag_wandb_entity

set -l wandb_project afa-team
set -ql _flag_wandb_project
and set wandb_project $_flag_wandb_project

# --- DATA GENERATION ---
./scripts/pipelines/generate_data.fish --dataset=cube --dataset=MNIST --split=1 --split=2 --launcher=$launcher --output-alias=$alias --wandb-entity=$wandb_entity --wandb-project=$wandb_project

# # --- PRE-TRAINING ---
./scripts/pipelines/shim2018/pretrain_shim2018.fish --dataset=cube --dataset=MNIST --split=1 --split=2 --launcher=$launcher --device=$device --dataset-alias=$alias --output-alias=$alias --wandb-entity=$wandb_entity --wandb-project=$wandb_project

# --- METHOD TRAINING ---
./scripts/pipelines/shim2018/train_shim2018.fish --dataset=cube --dataset=MNIST --budgets=3,5,10 --budgets=10,50,100 --split=1 --split=2 --launcher=$launcher --device=$device --speed=fast --pretrain-alias=$alias --output-alias=$alias --wandb-entity=$wandb_entity --wandb-project=$wandb_project

# --- CLASSIFIER TRAINING ---
./scripts/pipelines/train_classifier.fish --dataset=cube --dataset=MNIST --split=1 --split=2 --launcher=$launcher --device=$device --speed=fast --dataset-alias=$alias --output-alias=$alias --wandb-entity=$wandb_entity --wandb-project=$wandb_project

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
          eval_only_n_samples=10 \
          trained_method_artifact_name="(string join , $trained_method_artifact_names)" \
          trained_classifier_artifact_name=masked_mlp_classifier-$dataset""_split_$split:$alias,null hydra/launcher=$launcher"
        set i (math "$i+1")
    end
end

# --- PLOTTING ---
./scripts/pipelines/plot.fish --dataset=cube --dataset=MNIST --budgets=3,5,10 --budgets=10,50,100 --method=shim2018 --split=1 --split=2 --evaluation-alias=$alias --wandb-entity=$wandb_entity --wandb-project=$wandb_project
