#!/usr/bin/env bash

set -u  # Treat unset variables as an error
set -o pipefail  # Fail if any part of a pipeline fails

LAUNCHER=$1 # 'custom_slurm' or 'basic'
DEVICE=$2 # 'cuda' or 'cpu'

# run_parallel() {
#   local cmds=("$@")
#   local pids=()
#
#   for cmd in "${cmds[@]}"; do
#     echo "Launching: $cmd"
#     bash -c "$cmd" &
#     pids+=($!)
#   done
#
#   local i=0
#   for pid in "${pids[@]}"; do
#     wait "$pid" || {
#       echo "Command failed: ${cmds[$i]}"
#       exit 1
#     }
#     ((i++))
#   done
# }

# Ensure that $1 and $2 are set
if [ -z "$LAUNCHER" ] || [ -z "$DEVICE" ]; then
  echo "Usage: $0 <launcher> <device>"
  echo "Example: $0 custom_slurm cuda"
  exit 1
fi

EXTRA_OPTS="device=$DEVICE hydra/launcher=$LAUNCHER"

# --- DATA GENERATION ---
echo "Starting data generation job..."
sleep 1
# uv run scripts/data_generation/generate_dataset.py -m dataset=cube,MNIST split_idx=1,2 output_artifact_aliases=["example"] hydra/launcher=$LAUNCHER

# --- PRE-TRAINING ---
echo "Starting pretraining jobs..."
sleep 1

# Pretrain on cube data:
job1="uv run scripts/pretrain_models/pretrain_shim2018.py -m output_artifact_aliases=[\"example\"] dataset@_global_=cube_fast dataset_artifact_name=cube_split_1:example,cube_split_2:example $EXTRA_OPTS"

# Pretrain on MNIST data:
job2="uv run scripts/pretrain_models/pretrain_shim2018.py -m output_artifact_aliases=[\"example\"] dataset@_global_=MNIST_fast dataset_artifact_name=MNIST_split_1:example,MNIST_split_2:example $EXTRA_OPTS"

# mprocs "$job1" "$job2"

# --- METHOD TRAINING ---
echo "Starting method training jobs..."
sleep 1

# Train on the cube dataset:
job1="uv run scripts/train_methods/train_shim2018.py -m output_artifact_aliases=[\"example\"] dataset@_global_=cube_fast pretrained_model_artifact_name=pretrain_shim2018-cube_split_1:example,pretrain_shim2018-cube_split_2:example hard_budget=5,10 $EXTRA_OPTS"

# Train on the MNIST dataset:
job2="uv run scripts/train_methods/train_shim2018.py -m output_artifact_aliases=[\"example\"] dataset@_global_=MNIST_fast pretrained_model_artifact_name=pretrain_shim2018-MNIST_split_1:example,pretrain_shim2018-MNIST_split_2:example hard_budget=10,50 $EXTRA_OPTS"

# mprocs "$job1" "$job2"

# --- CLASSIFIER TRAINING ---
echo "Starting classifier training jobs..."
sleep 1

# Train on the cube dataset:
job1="uv run scripts/train_classifiers/train_masked_mlp_classifier.py -m output_artifact_aliases=[\"example\"] dataset@_global_=cube_fast dataset_artifact_name=cube_split_1:example,cube_split_2:example $EXTRA_OPTS"

# Train on the MNIST dataset:
job2="uv run scripts/train_classifiers/train_masked_mlp_classifier.py -m output_artifact_aliases=[\"example\"] dataset@_global_=MNIST_fast dataset_artifact_name=MNIST_split_1:example,MNIST_split_2:example $EXTRA_OPTS"

# mprocs "$job1" "$job2"

# --- EVALUATION ---
echo "Starting evaluation jobs..."
sleep 1

# cube, split 1, budget 5 & 10, external classifier and built-in classifier (null):
job1="uv run scripts/evaluation/eval_afa_method.py -m \
  output_artifact_aliases=[\"example\"] \
  eval_only_n_samples=100 \
  trained_method_artifact_name=train_shim2018-cube_split_1-budget_5-seed_42:example,train_shim2018-cube_split_1-budget_10-seed_42:example \
  trained_classifier_artifact_name=masked_mlp_classifier-cube_split_1:example,null hydra/launcher=$LAUNCHER"

# cube, split 2, budget 5 & 10, external classifier and built-in classifier (null):
job2="uv run scripts/evaluation/eval_afa_method.py -m \
  output_artifact_aliases=[\"example\"] \
  eval_only_n_samples=100 \
  trained_method_artifact_name=train_shim2018-cube_split_2-budget_5-seed_42:example,train_shim2018-cube_split_2-budget_10-seed_42:example \
  trained_classifier_artifact_name=masked_mlp_classifier-cube_split_2:example,null hydra/launcher=$LAUNCHER"

# MNIST, split 1, budget 10 & 50, external classifier and built-in classifier (null):
job3="uv run scripts/evaluation/eval_afa_method.py -m \
  output_artifact_aliases=[\"example\"] \
  eval_only_n_samples=100 \
  trained_method_artifact_name=train_shim2018-MNIST_split_1-budget_10-seed_42:example,train_shim2018-MNIST_split_1-budget_50-seed_42:example \
  trained_classifier_artifact_name=masked_mlp_classifier-MNIST_split_1:example,null hydra/launcher=$LAUNCHER"

# MNIST, split 2, budget 10 & 50, external classifier and built-in classifier (null):
job4="uv run scripts/evaluation/eval_afa_method.py -m \
  output_artifact_aliases=[\"example\"] \
  eval_only_n_samples=100 \
  trained_method_artifact_name=train_shim2018-MNIST_split_2-budget_10-seed_42:example,train_shim2018-MNIST_split_2-budget_50-seed_42:example \
  trained_classifier_artifact_name=masked_mlp_classifier-MNIST_split_2:example,null hydra/launcher=$LAUNCHER"

# mprocs "$job1" "$job2" "$job3" "$job4"

# --- PLOTTING ---
echo "Starting plotting job..."
sleep 1
uv run scripts/plotting/plot_results.py eval_artifact_names=[\
train_shim2018-cube_split_1-budget_5-seed_42-masked_mlp_classifier-cube_split_1:example,\
train_shim2018-cube_split_2-budget_5-seed_42-masked_mlp_classifier-cube_split_2:example,\
train_shim2018-cube_split_1-budget_10-seed_42-masked_mlp_classifier-cube_split_1:example,\
train_shim2018-cube_split_2-budget_10-seed_42-masked_mlp_classifier-cube_split_2:example,\
train_shim2018-cube_split_1-budget_5-seed_42-builtin:example,\
train_shim2018-cube_split_2-budget_5-seed_42-builtin:example,\
train_shim2018-cube_split_1-budget_10-seed_42-builtin:example,\
train_shim2018-cube_split_2-budget_10-seed_42-builtin:example,\
train_shim2018-MNIST_split_1-budget_10-seed_42-masked_mlp_classifier-MNIST_split_1:example,\
train_shim2018-MNIST_split_2-budget_10-seed_42-masked_mlp_classifier-MNIST_split_2:example,\
train_shim2018-MNIST_split_1-budget_50-seed_42-masked_mlp_classifier-MNIST_split_1:example,\
train_shim2018-MNIST_split_2-budget_50-seed_42-masked_mlp_classifier-MNIST_split_2:example,\
train_shim2018-MNIST_split_1-budget_10-seed_42-builtin:example,\
train_shim2018-MNIST_split_2-budget_10-seed_42-builtin:example,\
train_shim2018-MNIST_split_1-budget_50-seed_42-builtin:example,\
train_shim2018-MNIST_split_2-budget_50-seed_42-builtin:example]

