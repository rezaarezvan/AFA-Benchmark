#!/usr/bin/env bash

# cube
# uv run scripts/evaluation/eval_afa_method.py -m \
#   output_artifact_aliases=["Jun08"] \
#   trained_method_artifact_name="\
#           train_shim2018-cube_split_1-budget_3-seed_42:Jun06Nightly,\
#           train_shim2018-cube_split_1-budget_5-seed_42:Jun06Nightly,\
#           train_shim2018-cube_split_1-budget_10-seed_42:Jun06Nightly"\
#   trained_classifier_artifact_name="\
#           masked_mlp_classifier-cube_split_1:Jun07Nightly,\
#           null"
uv run scripts/evaluation/eval_afa_method.py -m \
  output_artifact_aliases=["Jun08"] \
  trained_method_artifact_name="\
          train_shim2018-cube_split_2-budget_3-seed_42:Jun06Nightly,\
          train_shim2018-cube_split_2-budget_5-seed_42:Jun06Nightly,\
          train_shim2018-cube_split_2-budget_10-seed_42:Jun06Nightly"\
  trained_classifier_artifact_name="\
          masked_mlp_classifier-cube_split_2:Jun07Nightly,\
          null"
uv run scripts/evaluation/eval_afa_method.py -m \
  output_artifact_aliases=["Jun08"] \
  trained_method_artifact_name="\
          train_shim2018-cube_split_3-budget_3-seed_42:Jun06Nightly,\
          train_shim2018-cube_split_3-budget_5-seed_42:Jun06Nightly,\
          train_shim2018-cube_split_3-budget_10-seed_42:Jun06Nightly"\
  trained_classifier_artifact_name="\
          masked_mlp_classifier-cube_split_3:Jun07Nightly,\
          null"
uv run scripts/evaluation/eval_afa_method.py -m \
  output_artifact_aliases=["Jun08"] \
  trained_method_artifact_name="\
          train_shim2018-cube_split_4-budget_3-seed_42:Jun06Nightly,\
          train_shim2018-cube_split_4-budget_5-seed_42:Jun06Nightly,\
          train_shim2018-cube_split_4-budget_10-seed_42:Jun06Nightly"\
  trained_classifier_artifact_name="\
          masked_mlp_classifier-cube_split_4:Jun07Nightly,\
          null"
uv run scripts/evaluation/eval_afa_method.py -m \
  output_artifact_aliases=["Jun08"] \
  trained_method_artifact_name="\
          train_shim2018-cube_split_5-budget_3-seed_42:Jun06Nightly,\
          train_shim2018-cube_split_5-budget_5-seed_42:Jun06Nightly,\
          train_shim2018-cube_split_5-budget_10-seed_42:Jun06Nightly"\
  trained_classifier_artifact_name="\
          masked_mlp_classifier-cube_split_5:Jun07Nightly,\
          null"

# MNIST
uv run scripts/evaluation/eval_afa_method.py -m \
  output_artifact_aliases=["Jun08"] \
  trained_method_artifact_name="\
          train_shim2018-MNIST_split_1-budget_3-seed_42:Jun06Nightly,\
          train_shim2018-MNIST_split_1-budget_5-seed_42:Jun06Nightly,\
          train_shim2018-MNIST_split_1-budget_10-seed_42:Jun06Nightly"\
  trained_classifier_artifact_name="\
          masked_mlp_classifier-MNIST_split_1:Jun07Nightly,\
          null"
uv run scripts/evaluation/eval_afa_method.py -m \
  output_artifact_aliases=["Jun08"] \
  trained_method_artifact_name="\
          train_shim2018-MNIST_split_2-budget_3-seed_42:Jun06Nightly,\
          train_shim2018-MNIST_split_2-budget_5-seed_42:Jun06Nightly,\
          train_shim2018-MNIST_split_2-budget_10-seed_42:Jun06Nightly"\
  trained_classifier_artifact_name="\
          masked_mlp_classifier-MNIST_split_2:Jun07Nightly,\
          null"
uv run scripts/evaluation/eval_afa_method.py -m \
  output_artifact_aliases=["Jun08"] \
  trained_method_artifact_name="\
          train_shim2018-MNIST_split_3-budget_3-seed_42:Jun06Nightly,\
          train_shim2018-MNIST_split_3-budget_5-seed_42:Jun06Nightly,\
          train_shim2018-MNIST_split_3-budget_10-seed_42:Jun06Nightly"\
  trained_classifier_artifact_name="\
          masked_mlp_classifier-MNIST_split_3:Jun07Nightly,\
          null"
uv run scripts/evaluation/eval_afa_method.py -m \
  output_artifact_aliases=["Jun08"] \
  trained_method_artifact_name="\
          train_shim2018-MNIST_split_4-budget_3-seed_42:Jun06Nightly,\
          train_shim2018-MNIST_split_4-budget_5-seed_42:Jun06Nightly,\
          train_shim2018-MNIST_split_4-budget_10-seed_42:Jun06Nightly"\
  trained_classifier_artifact_name="\
          masked_mlp_classifier-MNIST_split_4:Jun07Nightly,\
          null"
uv run scripts/evaluation/eval_afa_method.py -m \
  output_artifact_aliases=["Jun08"] \
  trained_method_artifact_name="\
          train_shim2018-MNIST_split_5-budget_3-seed_42:Jun06Nightly,\
          train_shim2018-MNIST_split_5-budget_5-seed_42:Jun06Nightly,\
          train_shim2018-MNIST_split_5-budget_10-seed_42:Jun06Nightly"\
  trained_classifier_artifact_name="\
          masked_mlp_classifier-MNIST_split_5:Jun07Nightly,\
          null"
