# Generate data
uv run scripts/generate_dataset.py -m dataset=cube,MNIST split_idx=1,2 output_artifact_aliases=["example"]

# --- PRE-TRAINING ---
# Pretrain on cube data:
job1='uv run scripts/pretrain_models/pretrain_shim2018.py -m output_artifact_aliases=["example"] dataset@_global_=cube_fast dataset_artifact_name=cube_split_1:example,cube_split_2:example'

# Pretrain on MNIST data:
job2='uv run scripts/pretrain_models/pretrain_shim2018.py -m output_artifact_aliases=["example"] dataset@_global_=MNIST_fast dataset_artifact_name=MNIST_split_1:example,MNIST_split_2:example'

mprocs "$job1" "$job2"

# --- METHOD TRAINING ---
# Train on the cube dataset:
job1 = 'uv run scripts/train_methods/train_shim2018.py -m output_artifact_aliases=["example"] dataset@_global_=cube_fast pretrained_model_artifact_name=pretrain_shim2018-cube_split_1:example,pretrain_shim2018-cube_split_2:example hard_budget=5,10'

# Train on the MNIST dataset:
job2 = 'uv run scripts/train_methods/train_shim2018.py -m output_artifact_aliases=["example"] dataset@_global_=MNIST_fast pretrained_model_artifact_name=pretrain_shim2018-MNIST_split_1:example,pretrain_shim2018-MNIST_split_2:example hard_budget=10,50'

mprocs job1 job2

# --- CLASSIFIER TRAINING ---
# Train on the cube dataset:
job1 = 'uv run scripts/train_classifiers/train_masked_mlp_classifier.py -m output_artifact_aliases=["example"] dataset@_global_=cube_fast dataset_artifact_name=cube_split_1:example,cube_split_2:example'

# Train on the MNIST dataset:
job2 = 'uv run scripts/train_classifiers/train_masked_mlp_classifier.py -m output_artifact_aliases=["example"] dataset@_global_=MNIST_fast dataset_artifact_name=MNIST_split_1:example,MNIST_split_2:example'

mprocs job1 job2

# --- EVALUATION ---
# CUBE, split 1, budget 5 & 10, external classifier and built-in classifier (null):
job1 = 'uv run scripts/evaluation/eval_afa_method.py -m \
  output_artifact_aliases=["example"] \
  eval_only_n_samples=100 \
  trained_method_artifact_name=train_shim2018-cube_split_1-budget_5-seed_42:example,train_shim2018-cube_split_1-budget_10-seed_42:example \
  trained_classifier_artifact_name=masked_mlp_classifier-cube_split_1:example,null'

# CUBE, split 2, budget 5 & 10, external classifier and built-in classifier ():
job2 = 'uv run scripts/evaluation/eval_afa_method.py -m \
  output_artifact_aliases=["example"] \
  eval_only_n_samples=100 \
  trained_method_artifact_name=train_shim2018-cube_split_2-budget_5-seed_42:example,train_shim2018-cube_split_2-budget_10-seed_42:example \
  trained_classifier_artifact_name=masked_mlp_classifier-cube_split_2:example,null'

# TODO: remaining jobs
