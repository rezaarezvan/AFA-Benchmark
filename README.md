# AFA-Benchmark

Methods to compare:

Generative greedy:

1. Chao Ma, et al. Eddi: Efficient dynamic discovery of high-value information with partial VAE. ICML 2019.
2. Wenbo Gong, et al. Icebreaker: Element-wise efficient information acquisition with a bayesian deep latent gaussian model. NeurIPS 2019.

Discriminative greedy:

3. Ian Connick Covert, et al. Learning to maximize mutual information for dynamic feature selection. ICML 2023.
4. Soham Gadgil, et al. Estimating conditional mutual information for dynamic feature selection. ICML 2024.

RL:

5. Hajin Shim, et al. Joint active feature acquisition and classification with variable-size set
encoding. NeurIPS 2018.
6. Jaromír Janisch, et al. Classification with costly features using deep reinforcement learning. AAAI 2019.
7. Sara Zannone, et al. Odin: Optimal discovery of high-value information using model-based deep reinforcement learning. ICML workshop, 2019.
8. Yang Li and Junier Oliva. Active feature acquisition with generative surrogate models. ICML 2021.


Maybe:

9. Mohammad Kachuee, et al. Opportunistic learning: Budgeted cost-sensitive learning from data streams. ICLR, 2019.)
10. Gabriel Dulac-Arnold, et al. Datum-wise classification: a sequential approach to sparsity. ECML PKDD 2011.
11. Thomas Rückstieß, et al. Sequential feature selection for classification. AI 2011.
12. Yang Li and Junier Oliva. Distribution guided active feature acquisition, arxiv 2024. **(Journal extension of Li and Oliva 2021).**
13. He He, et al. Imitation learning by coaching. NeurIPS 2012
14. Jaromír Janisch, et al. Classification with costly features as a sequential decision-making problem. MLJ 2020. **(Journal extension of Janisch et al 2019).**
15. Aditya Chattopadhyay, et al. Variational information pursuit for interpretable predictions. ICLR, 2023.
16. Samrudhdhi B Rangrej et al. A probabilistic hard attention model for sequentially observed scenes. BMVC 2021. **(Extends EDDI to image data).**
17. Ghosh et al. DiFA: Differentiable Feature Acquisition. AAAI 2023.
18. Valancius et al. Acquisition conditioned oracle for nongreedy active feature acquisition, ICML 2024.


## State requirements per method at test time

All of the four RL methods above need two things at test time:
- Currently selected features as a vector, with $0$ for non-acquired features.
- Boolean feature mask. $1$ if feature is acquired, $0$ if not.

## Synthetic dataset generation

Things to consider for synthetic data generation in the context of AFA (extension of CUBE):

- Number of features.
- Number of classes.
- Number of informative/redundant features for each class.
- Degree of overlap in which features are relevant for each class.
- Cost profiles (uniform vs. highly skewed).
- Label noise
- Class balance
- Synthetic data where non-greedy selection is better than greedy

## Setup

Dependencies are handled using [uv](https://docs.astral.sh/uv/). Install uv and then run `uv sync` in the root directory.

## Terminology

- AFA method / method: An algorithm that sequentially selects features to acquire, based on the current state of acquired features and the data. Is usually also able to make label predictions in each step.
- Model: A neural network.
- Dataset/method/classifier type: "Type" refers to strings that can be passed to the functions in the `registry` module.

## Pipeline parts

### 1. Data generation

Generate data with `uv run scripts/generate_data.py`. This will place the data in the `data` directory but also upload it to wandb. By default, the dataset artifacts are saved with names in the format `<dataset_type>_split_<split_idx>`, e.g `MNIST_SPLIT_1`. You can add custom aliases using the `--output_artifact_aliases` argument.

### 2. Pre-training

This phase is only relevant for AFA methods which require pre-training of some model, such as the RL methods. Since this phase can be different for each method, there is no common interface.

### 3. Method training

Training is also method-dependent, but the resulting artifact has to have the same format, as explained [here](#trained-method-artifacts).

### 4. Training a classifier (optional)

During evaluation, it can be useful to use a common classifier to compare different AFA methods, in order to isolate the effect of the feature selection mechanism.

To train a MLP classifier, run `uv run scripts/train_classifiers/train_masked_classifier.py` which will use the configuration defined in `conf/classifiers/masked_mlp_classifier/tmp.yaml`.

### 5. Evaluation

Once the model has been trained (and optionally a classifier as well), you can evaluate it using `uv run scripts/eval_afa_method.py`. This will use the configuration defined in `conf/eval/tmp.yaml`, where you can also specify whether an external classifier should be used. The evaluation results will be saved as an artifact with the metadata keys described [here](#evaluation-artifacts).

Note that the same evaluation script is used for all AFA methods, you just have to point the configuration artifact name to the correct method.

### 6. Plotting results

## Artifact structure

### Dataset artifacts

Each dataset artifact is expected to contain a `dataset_type` key in its metadata. Passing this string to the `get_afa_dataset_class` function should return the correct dataset class. The `load` method of the dataset class is then called on the files "train.pt", "val.pt" and "test.pt" in the artifact. See `load_dataset_artifact` for more info.

### Trained method artifacts

Each trained method should have the following keys in its metadata:
- `afa_method_class` (`str`): A string that can be passed to the `get_afa_method_class` function to return the correct AFA method class. The `load` method of this class is then called with the contents of the artifact. See `load_trained_method_artifact` for more info.
- `budget`` (`int|None`): Which budget this method was trained with. For some methods, this is not applicable, in which case it should be set to `None`.
- `dataset_artifact_name` (`str`): Which artifact this method was trained on. This is used to load the same dataset artifact with `load_dataset_artifact` during evaluation.
- `dataset_type` (`str`): The dataset type of `dataset_artifact_name`. Superfluous but convenient.
- `method_type` (`str`): The method type of the trained afa method.
- `seed` (`int`): The random seed used for training this method.

Note that `afa_method_class` can be the same for several different `method_type`s. This is the case for the RL methods which have a common AFA method class `RLAFAMethod` but have different `method_type`s like `"shim2018"` and `"zannone2019"` in order to distinguish them during evaluation.

### Trained classifier artifacts

TODO

### Evaluation artifacts

The `eval_afa_method.py` script generates artifacts with the following keys in their metadata:
- `budget` (`int`): The budget used for evaluation. If the trained method artifact had a budget set, the same budget is used here. If the trained method artifact had no budget set, the budget is equal to the number of features in the dataset.
- `classifier_type` (`str`): The classifier type used for evaluation. Either a "normal" classifier type that can be passed to `get_afa_classifier_class`, or `"builtin"`, in which case the built-in classifier of the trained method was used.
- `dataset_type` (`str`): The dataset type of the dataset used for evaluation. Always the same as the `dataset_type` of the trained method artifact.
- `method_type` (`str`): The method type of the trained method artifact.
- `seed` (`int`): The random seed used for evaluation.

## Hydra

All scripts in this repository use [Hydra](https://hydra.cc/) for configuration management, mainly to simplify batching of experiments. [Structured configs](https://hydra.cc/docs/tutorials/structured_config/intro/) are defined in `src/common/config_classes.py` and the configurations themselves are defined in the `conf` directory.


## Full pipeline example using shim2018

Here is an example pipeline for evaluating the `shim2018` method with two different budgets and two different splits on two different datasets: `cube` and `MNIST`. We will use the artifact alias `"example"` in each step to ensure that we use the intended artifacts, and only use a epoch to speed up the process. For meaningful results, remove the `epochs=X` argument from the commands below to use the default value instead.

Hydra multiruns (using the `-m` flag) can be used with [Slurm](https://slurm.schedmd.com/) using the [Submitit](https://hydra.cc/docs/plugins/submitit_launcher/) plugin. This allows you to run jobs in parallel. In this case, append `hydra/launcher=submitit_slurm` to all the multirun commands below.

### 0. WandB setup

To log results to your own Weights & Biases project, set the `WANDB_PROJECT` environment variable to your project name. If you want to use a different Weights & Biases entity than the default one, set the `WANDB_ENTITY` environment variable to your entity name.

If you want to run everything locally, use `wandb local`.

### 1. Generate data

```bash
uv run scripts/generate_dataset.py -m dataset=cube,MNIST split_idx=1,2 output_artifact_aliases=["example"]
```

This produces the 4 dataset artifacts
- `cube_split_1:example`
- `cube_split_2:example`
- `MNIST_split_1:example`
- `MNIST_split_2:example`

### 2. Pre-training

```bash
uv run scripts/pretrain_methods/pretrain_shim2018.py -m output_artifact_aliases=["example"] epochs=1 dataset_artifact_name=cube_split_1:example,cube_split_2:example,MNIST_split_1:example,MNIST_split_2:example
```

This produces 4 pre-training artifacts:
- `pretrain_shim2018-cube_split_1:example`
- `pretrain_shim2018-cube_split_2:example`
- `pretrain_shim2018-MNIST_split_1:example`
- `pretrain_shim2018-MNIST_split_2:example`


### 3. Training the method

```bash
uv run scripts/train_methods/train_shim2018.py -m output_artifact_aliases=["example"] dataset@_global_=cube_fast pretrained_model_artifact_name=pretrain_shim2018-cube_split_1:example,pretrain_shim2018-cube_split_2:example hard_budget=5,10
```

```bash
uv run scripts/train_methods/train_shim2018.py -m output_artifact_aliases=["example"] dataset@_global_=MNIST_fast pretrained_model_artifact_name=pretrain_shim2018-MNIST_split_1:example,pretrain_shim2018-MNIST_split_2:example hard_budget=10,50
```

This produces 8 trained method artifacts, 4 per budget:
- `train_shim2018-cube_split_1-budget_5-seed_42:example`
- `train_shim2018-cube_split_2-budget_5-seed_42:example`
- `train_shim2018-MNIST_split_1:example-budget_10-seed_42:example`
- `train_shim2018-MNIST_split_2:example-budget_10-seed_42:example`
- `train_shim2018-cube_split_1-budget_10-seed_42:example`
- `train_shim2018-cube_split_2-budget_10-seed_42:example`
- `train_shim2018-MNIST_split_1:example-budget_50-seed_42:example`
- `train_shim2018-MNIST_split_2:example-budget_50-seed_42:example`

### 4. Training a classifier (optional)

This is similar to the pre-training step.

```bash
uv run scripts/train_classifiers/train_masked_mlp_classifier.py -m output_artifact_aliases=["example"] epochs=1 evaluate_final_performance=false dataset_artifact_name=cube_split_1:example,cube_split_2:example,MNIST_split_1:example,MNIST_split_2:example
```

This produces 4 classifier artifacts:
- `masked_mlp_classifier-cube_split_1:example`
- `masked_mlp_classifier-cube_split_2:example`
- `masked_mlp_classifier-MNIST_split_1:example`
- `masked_mlp_classifier-MNIST_split_2:example`

### 5. Evaluation

Since we do not want to mix models trained on different datasets, we have to use several commands, one per dataset/split combination.

CUBE, split 1, budget 5 & 10, external classifier and built-in classifier (null):
```bash
uv run scripts/eval_afa_method.py -m \
  output_artifact_aliases=["example"] \
  eval_only_n_samples=100 \
  trained_method_artifact_name=train_shim2018-cube_split_1-budget_5-seed_42:example,train_shim2018-cube_split_1-budget_10-seed_42:example \
  trained_classifier_artifact_name=masked_mlp_classifier-cube_split_1:example,null
```

CUBE, split 2, budget 5 & 10, external classifier and built-in classifier ():
```bash
uv run scripts/eval_afa_method.py -m \
  output_artifact_aliases=["example"] \
  eval_only_n_samples=100 \
  trained_method_artifact_name=train_shim2018-cube_split_2-budget_5-seed_42:example,train_shim2018-cube_split_2-budget_10-seed_42:example \
  trained_classifier_artifact_name=masked_mlp_classifier-cube_split_2:example,null
```

MNIST, split 1, budget 5 & 10, external classifier and built-in classifier (null):
```bash
uv run scripts/eval_afa_method.py -m \
  output_artifact_aliases=["example"] \
  eval_only_n_samples=100 \
  trained_method_artifact_name=train_shim2018-MNIST_split_1-budget_5-seed_42:example,train_shim2018-MNIST_split_1-budget_10-seed_42:example \
  trained_classifier_artifact_name=masked_mlp_classifier-MNIST_split_1:example,null
```

MNIST, split 2, budget 5 & 10, external classifier and built-in classifier (null):
```bash
uv run scripts/eval_afa_method.py -m \
  output_artifact_aliases=["example"] \
  eval_only_n_samples=100 \
  trained_method_artifact_name=train_shim2018-MNIST_split_2-budget_5-seed_42:example,train_shim2018-MNIST_split_2-budget_10-seed_42:example \
  trained_classifier_artifact_name=masked_mlp_classifier-MNIST_split_2:example,null
```

Or a single command that chains them together:
```bash
uv run scripts/eval_afa_method.py -m \
  output_artifact_aliases=["example"] \
  eval_only_n_samples=100 \
  trained_method_artifact_name=train_shim2018-cube_split_1-budget_5-seed_42:example,train_shim2018-cube_split_1-budget_10-seed_42:example \
  trained_classifier_artifact_name=masked_mlp_classifier-cube_split_1:example,null && \
uv run scripts/eval_afa_method.py -m \
  output_artifact_aliases=["example"] \
  eval_only_n_samples=100 \
  trained_method_artifact_name=train_shim2018-cube_split_2-budget_5-seed_42:example,train_shim2018-cube_split_2-budget_10-seed_42:example \
  trained_classifier_artifact_name=masked_mlp_classifier-cube_split_2:example,null && \
uv run scripts/eval_afa_method.py -m \
  output_artifact_aliases=["example"] \
  eval_only_n_samples=100 \
  trained_method_artifact_name=train_shim2018-MNIST_split_1-budget_5-seed_42:example,train_shim2018-MNIST_split_1-budget_10-seed_42:example \
  trained_classifier_artifact_name=masked_mlp_classifier-MNIST_split_1:example,null && \
uv run scripts/eval_afa_method.py -m \
  output_artifact_aliases=["example"] \
  eval_only_n_samples=100 \
  trained_method_artifact_name=train_shim2018-MNIST_split_2-budget_5-seed_42:example,train_shim2018-MNIST_split_2-budget_10-seed_42:example \
  trained_classifier_artifact_name=masked_mlp_classifier-MNIST_split_2:example,null
```

This produces 16 evaluation artifacts in total:
- `train_shim2018-cube_split_1-budget_5-seed_42-builtin:example`
- `train_shim2018-cube_split_1-budget_5-seed_42-masked_mlp_classifier-cube_split_1:example`
- `train_shim2018-cube_split_1-budget_10-seed_42-builtin:example`
- `train_shim2018-cube_split_1-budget_10-seed_42-masked_mlp_classifier-cube_split_1:example`
- `train_shim2018-cube_split_2-budget_5-seed_42-builtin:example`
- `train_shim2018-cube_split_2-budget_5-seed_42-masked_mlp_classifier-cube_split_2:example`
- `train_shim2018-cube_split_2-budget_10-seed_42-builtin:example`
- `train_shim2018-cube_split_2-budget_10-seed_42-masked_mlp_classifier-cube_split_2:example`
- `train_shim2018-MNIST_split_1-budget_5-seed_42-builtin:example`
- `train_shim2018-MNIST_split_1-budget_5-seed_42-masked_mlp_classifier-MNIST_split_1:example`
- `train_shim2018-MNIST_split_1-budget_10-seed_42-builtin:example`
- `train_shim2018-MNIST_split_1-budget_10-seed_42-masked_mlp_classifier-MNIST_split_1:example`
- `train_shim2018-MNIST_split_2-budget_5-seed_42-builtin:example`
- `train_shim2018-MNIST_split_2-budget_5-seed_42-masked_mlp_classifier-MNIST_split_2:example`
- `train_shim2018-MNIST_split_2-budget_10-seed_42-builtin:example`
- `train_shim2018-MNIST_split_2-budget_10-seed_42-masked_mlp_classifier-MNIST_split_2:example`

### 6. Plotting results

We can now aggregate the results in plots:

## How to extend

### Adding a new dataset

TODO

### Adding a new AFA method

TODO

### Adding a new classifier

TODO

### Changing the evaluation
