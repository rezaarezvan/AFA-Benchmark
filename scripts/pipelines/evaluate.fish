#!/usr/bin/env fish

# set -g fish_trace 1

# -----------------------
# Parse args
# -----------------------

argparse "help" "batch_size=?" "device=?" "launcher=?" "output-alias=?" "wandb-entity=?" "wandb-project=?" -- $argv
or exit 1

# Print help if specified
if set -ql _flag_help
    echo "Usage: evaluate.fish [batch_size=<int>] [--device={cpu,cuda}] [--help] [--launcher={custom_slurm,basic}] [--output-alias=<str>] [--wandb-entity=<str>] [--wandb-project=<str>]" >&2
    exit 1
end

# Default arguments

set -g batch_size 128
set -q _flag_batch_size
and set batch_size $_flag_batch_size

set -g device cuda
set -q _flag_device
and set device $_flag_device

set -g launcher custom_slurm
set -q _flag_launcher
and set launcher $_flag_launcher

set -g output_alias tmp
set -q _flag_output_alias
and set output_alias $_flag_output_alias

set -l wandb_entity afa-team
set -q _flag_wandb_entity
and set wandb_entity $_flag_wandb_entity

set -l wandb_project afa-benchmark
set -q _flag_wandb_project
and set wandb_project $_flag_wandb_project

set -gx WANDB_ENTITY $wandb_entity
set -gx WANDB_PROJECT $wandb_project

set -g extra_opts "hydra/launcher=$launcher"

function build_eval_job
    # Args:
    # 1: trained_method_artifact_names: str (comma separated)
    # 2: trained_classifier_artifact_name: str
    echo "uv run scripts/evaluation/eval_afa_method.py -m \
      output_artifact_aliases=[\"$output_alias\"] \
      trained_method_artifact_name=$argv[1] \
      trained_classifier_artifact_name=$argv[2],null \
      batch_size=$batch_size \
      device=$device \
      $extra_opts"
end

set -l classifier_alias Jun11Slow-a

# cube, split 1
set -l trained_method_artifact_names \
    train_shim2018-cube_split_1-budget_3-seed_42:Jul03a \
    train_shim2018-cube_split_1-budget_5-seed_42:Jul03a \
    train_shim2018-cube_split_1-budget_10-seed_42:Jul03a \
    # train_randomdummy-cube_split_1-budget_3-seed_42:Jun11Slow-a \
    # train_randomdummy-cube_split_1-budget_5-seed_42:Jun11Slow-a \
    # train_randomdummy-cube_split_1-budget_10-seed_42:Jun11Slow-a
set trained_method_artifact_names (string join , $trained_method_artifact_names)
set -l trained_classifier_artifact_name masked_mlp_classifier-cube_split_1:$classifier_alias
set -l job1 (build_eval_job $trained_method_artifact_names $trained_classifier_artifact_name)

# cube, split 2
set -l trained_method_artifact_names \
    train_shim2018-cube_split_2-budget_3-seed_42:Jul03a \
    train_shim2018-cube_split_2-budget_5-seed_42:Jul03a \
    train_shim2018-cube_split_2-budget_10-seed_42:Jul03a \
    # train_randomdummy-cube_split_2-budget_3-seed_42:Jun11Slow-a \
    # train_randomdummy-cube_split_2-budget_5-seed_42:Jun11Slow-a \
    # train_randomdummy-cube_split_2-budget_10-seed_42:Jun11Slow-a
set trained_method_artifact_names (string join , $trained_method_artifact_names)
set -l trained_classifier_artifact_name masked_mlp_classifier-cube_split_2:$classifier_alias
set -l job2 (build_eval_job $trained_method_artifact_names $trained_classifier_artifact_name)

# MNIST, split 1
set -l trained_method_artifact_names \
    train_shim2018-MNIST_split_1-budget_10-seed_42:Jul03a \
    train_shim2018-MNIST_split_1-budget_50-seed_42:Jul03a \
    train_shim2018-MNIST_split_1-budget_100-seed_42:Jul03a \
    # train_randomdummy-MNIST_split_1-budget_10-seed_42:Jun11Slow-a \
    # train_randomdummy-MNIST_split_1-budget_50-seed_42:Jun11Slow-a \
    # train_randomdummy-MNIST_split_1-budget_100-seed_42:Jun11Slow-a
set trained_method_artifact_names (string join , $trained_method_artifact_names)
set -l trained_classifier_artifact_name masked_mlp_classifier-MNIST_split_1:$classifier_alias
set -l job3 (build_eval_job $trained_method_artifact_names $trained_classifier_artifact_name)

# MNIST, split 2
set -l trained_method_artifact_names \
    train_shim2018-MNIST_split_2-budget_10-seed_42:Jul03a \
    train_shim2018-MNIST_split_2-budget_50-seed_42:Jul03a \
    train_shim2018-MNIST_split_2-budget_100-seed_42:Jul03a \
    # train_randomdummy-MNIST_split_2-budget_10-seed_42:Jun11Slow-a \
    # train_randomdummy-MNIST_split_2-budget_50-seed_42:Jun11Slow-a \
    # train_randomdummy-MNIST_split_2-budget_100-seed_42:Jun11Slow-a
set trained_method_artifact_names (string join , $trained_method_artifact_names)
set -l trained_classifier_artifact_name masked_mlp_classifier-MNIST_split_2:$classifier_alias
set -l job4 (build_eval_job $trained_method_artifact_names $trained_classifier_artifact_name)

# Launch all jobs
mprocs "$job1" "$job2" "$job3" "$job4"
