#!/usr/bin/env bash
set -euo pipefail

# -----------------------
# Parse args
# -----------------------
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <launcher> <device> <speed> [output_alias] [steps...]"
    echo "Steps: 1=Generation, 2=Pretrain, 3=Method, 4=Classifier, 5=Eval, 6=Plotting"
    echo "Example: $0 slurm cuda slow myalias 2 4 5"
    exit 1
fi

launcher="$1"
device="$2"
speed="$3"
output_alias="${4:-example}"
shift 4 || true
step_args=("${@:-1 2 3 4 5 6}")

# Validate speed
if [[ "$speed" == "slow" ]]; then
    dataset_suffix=""
elif [[ "$speed" == "fast" ]]; then
    dataset_suffix="_fast"
else
    echo "Third argument should either be 'slow' or 'fast'"
    exit 1
fi

extra_opts="device=$device hydra/launcher=$launcher"

# -----------------------
# Enable steps
# -----------------------
step_enabled() {
    for step in "${step_args[@]}"; do
        [[ $step -eq $1 ]] && return 0
    done
    return 1
}

RUN_GENERATION=$(step_enabled 1 && echo true || echo false)
RUN_PRETRAIN=$(step_enabled 2 && echo true || echo false)
RUN_METHOD=$(step_enabled 3 && echo true || echo false)
RUN_CLASSIFIER=$(step_enabled 4 && echo true || echo false)
RUN_EVAL=$(step_enabled 5 && echo true || echo false)
RUN_PLOTTING=$(step_enabled 6 && echo true || echo false)

# -----------------------
# Pipeline
# -----------------------


# --- DATA GENERATION ---
echo "Starting data generation job..."
sleep 1
# uv run scripts/data_generation/generate_dataset.py -m dataset=cube,MNIST split_idx=1,2 output_artifact_aliases=["example"] hydra/launcher=$LAUNCHER

# --- PRE-TRAINING ---
echo "Starting pretraining jobs..."
sleep 1

# Pretrain on cube data:
job1="uv run scripts/pretrain_models/pretrain_shim2018.py -m output_artifact_aliases=[\"example\"] dataset@_global_=cube$dataset_suffix dataset_artifact_name=cube_split_1:example,cube_split_2:example $EXTRA_OPTS"

# Pretrain on MNIST data:
job2="uv run scripts/pretrain_models/pretrain_shim2018.py -m output_artifact_aliases=[\"example\"] dataset@_global_=MNIST$dataset_suffix dataset_artifact_name=MNIST_split_1:example,MNIST_split_2:example $EXTRA_OPTS"

mprocs "$job1" "$job2"

# --- METHOD TRAINING ---
echo "Starting method training jobs..."
sleep 1

# Train on the cube dataset:
job1="uv run scripts/train_methods/train_shim2018.py -m output_artifact_aliases=[\"example\"] dataset@_global_=cube$dataset_suffix pretrained_model_artifact_name=pretrain_shim2018-cube_split_1:example,pretrain_shim2018-cube_split_2:example hard_budget=5,10 $EXTRA_OPTS"

# Train on the MNIST dataset:
job2="uv run scripts/train_methods/train_shim2018.py -m output_artifact_aliases=[\"example\"] dataset@_global_=MNIST$dataset_suffix pretrained_model_artifact_name=pretrain_shim2018-MNIST_split_1:example,pretrain_shim2018-MNIST_split_2:example hard_budget=10,50 $EXTRA_OPTS"

# mprocs "$job1" "$job2"

# --- CLASSIFIER TRAINING ---
echo "Starting classifier training jobs..."
sleep 1

# Train on the cube dataset:
job1="uv run scripts/train_classifiers/train_masked_mlp_classifier.py -m output_artifact_aliases=[\"example\"] dataset@_global_=cube$dataset_suffix dataset_artifact_name=cube_split_1:example,cube_split_2:example $EXTRA_OPTS"

# Train on the MNIST dataset:
job2="uv run scripts/train_classifiers/train_masked_mlp_classifier.py -m output_artifact_aliases=[\"example\"] dataset@_global_=MNIST$dataset_suffix dataset_artifact_name=MNIST_split_1:example,MNIST_split_2:example $EXTRA_OPTS"

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
train_shim2018-cube_split_1-budget_5-seed_42-builtin:example,\
train_shim2018-cube_split_2-budget_5-seed_42-builtin:example,\
train_shim2018-cube_split_1-budget_10-seed_42-builtin:example,\
train_shim2018-cube_split_2-budget_10-seed_42-builtin:example]

# uv run scripts/plotting/plot_results.py eval_artifact_names=[\
# train_shim2018-cube_split_1-budget_5-seed_42-masked_mlp_classifier-cube_split_1:example,\
# train_shim2018-cube_split_2-budget_5-seed_42-masked_mlp_classifier-cube_split_2:example,\
# train_shim2018-cube_split_1-budget_10-seed_42-masked_mlp_classifier-cube_split_1:example,\
# train_shim2018-cube_split_2-budget_10-seed_42-masked_mlp_classifier-cube_split_2:example,\
# train_shim2018-cube_split_1-budget_5-seed_42-builtin:example,\
# train_shim2018-cube_split_2-budget_5-seed_42-builtin:example,\
# train_shim2018-cube_split_1-budget_10-seed_42-builtin:example,\
# train_shim2018-cube_split_2-budget_10-seed_42-builtin:example,\
# train_shim2018-MNIST_split_1-budget_10-seed_42-masked_mlp_classifier-MNIST_split_1:example,\
# train_shim2018-MNIST_split_2-budget_10-seed_42-masked_mlp_classifier-MNIST_split_2:example,\
# train_shim2018-MNIST_split_1-budget_50-seed_42-masked_mlp_classifier-MNIST_split_1:example,\
# train_shim2018-MNIST_split_2-budget_50-seed_42-masked_mlp_classifier-MNIST_split_2:example,\
# train_shim2018-MNIST_split_1-budget_10-seed_42-builtin:example,\
# train_shim2018-MNIST_split_2-budget_10-seed_42-builtin:example,\
# train_shim2018-MNIST_split_1-budget_50-seed_42-builtin:example,\
# train_shim2018-MNIST_split_2-budget_50-seed_42-builtin:example]
