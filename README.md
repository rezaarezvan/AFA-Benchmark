# AFA Benchmark

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: ](https://img.shields.io/badge/License-XYZ-yellow.svg)]()
[![Paper](https://img.shields.io/badge/KDD%202025-Paper-red.svg)]()
[![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg)]()

**A comprehensive benchmark for Active Feature Acquisition (AFA) methods**

Compare state-of-the-art algorithms for sequential feature selection in scenarios where acquiring features is costly. Includes implementations of multiple AFA methods, standardized datasets, and automated evaluation pipelines.

## Quick Start

```bash
# Install dependencies
pip install uv && uv sync

# Run a simple comparison
uv run scripts/examples/quick_demo.py --dataset cube --methods aaco,EDDI

# View results
open results/demo_plot.html
```

## What is Active Feature Acquisition?
**Active Feature Acquisition (AFA)** addresses scenarios where,

- **Features are expensive** to obtain (medical tests, surveys, sensors),
- **Real-time decisions** must be made with partial information,
- **Budget constraints** limit which features you can acquire.

TODO, give a better example

**Example**: Medical diagnosis where each test costs money and time. AFA methods intelligently decide which tests to order next based on previous results, aiming for accurate diagnosis with minimal cost.

**Visual Example**
```
Patient arrives → Blood test → More tests needed?
     ↓              ↓              ↓
  Age: 45      Glucose: High    → AFA decides: Check HbA1c
  Sex: M       Pressure: Normal → Skip expensive MRI
               ↓
           Diagnosis with 3 tests instead of 10
```

## Installation

### Prerequisites
- Python 3.12
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- [Weights & Biases](https://wandb.ai) account (for experiment tracking)

### Setup

```bash
# Clone repository
git clone https://github.com/Linusaronsson/AFA-Benchmark.git
cd AFA-Benchmark

# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install -e .

# Setup W&B (optional but recommended)
wandb login
```

Additionally, if you have access to a cluster running [slurm](https://slurm.schedmd.com/), you might be interested in adding a configuration file to the `conf/global/hydra/launcher/` directory. The name of this file can then be referenced in scripts in order to run experiments in parallel.

## Simple Example

Train and evaluate a single AFA method on synthetic data:

```bash
# 1. Generate synthetic dataset
uv run scripts/data_generation/generate_data.py \
  dataset_type=cube \
  +output_artifact_aliases=[demo]

# 2. Train AACO method
uv run scripts/train_methods/train_aaco.py \
  dataset_artifact_name="cube_split_1:demo"

# 3. Evaluate performance
uv run scripts/evaluation/eval_afa_method.py \
  trained_method_artifact_name="aaco-cube_split_1:latest"

# 4. Generate plots
uv run scripts/plotting/plot_results.py \
  eval_artifact_names=["eval-aaco-cube_split_1:latest"]
```

**Expected output**: Plots showing accuracy vs. number of features acquired, saved to W&B and locally.

## Implemented Methods

| Method | Paper | Strategy | Greedy? |
|:--------:|:--------:|:--------:|:--------:|
| **EDDI-GG** | [link](https://proceedings.mlr.press/v97/ma19c) | Generative estimation of CMI | Greedy |
| **GDFS-DG** | [link](https://proceedings.mlr.press/v202/covert23a) | Discriminative estimation of CMI | Greedy |
| **DIME-DG** | [link](https://arxiv.org/pdf/2306.03301) | Discriminative estimation of CMI | Greedy |
| **JAFA-MFRL** | [link]() | Model-free RL | Non-greedy |
| **OL-MFRL** | [link]() | Model-free RL | Non-greedy |
| **ODIN-MFRL** | [link]() | Model-free RL | Non-greedy |
| **ODIN-MBRL** | [link]() | Model-based RL | Non-greedy |
| **AACO** | [link]() | Oracle-based | Non-greedy |
| **PT-S** | [link](https://link.springer.com/article/10.1023/A:1010933404324) | Global feature importance | N/A |
| **CAE-S** | [link](https://proceedings.mlr.press/v97/balin19a.html) | Global feature importance | N/A |

## Datasets

| Dataset | Type | Size (total, # data instances) | # Features | # Classes |
|:---------:|:----------:|:---------:|:---------:|:----------:|
| **CUBE** | Synthetic | 1000 | 20 | 8 |
| **AFAContext** | Synthetic | 1000 | 30 | 8 |
| **MNIST** | Real World | 60 000| 784 | 10 |
| **FashionMNIST** | Real World |  |  |  |
| **Diabetes** | Real World | 92 063 | 45 | 3 |
| **miniboone** | Real World |  |  |  |
| **Physionet** | Real World | 12 000 | 41 | 2 |

## Full Pipeline Tutorial TODO CHECK IF THIS WORKS

Here follows an example for training **ODIN**, one of the RL methods implemented in this benchmark.

### Dataset generation
First, generate some data. You can choose hyperparameters by creating new configurations in `conf/dataset_generation/dataset/`. Let's assume that you want to generate two noiseless versions of the **cube** and **AFAContext** datasets. Then run
```bash
uv run scripts/dataset_generation/generate_dataset.py -m hydra/launcher=submitit_basic dataset=AFAContext_without_noise,cube_without_noise split_idx=1,2 output_artifact_aliases=["tutorial-data"]
```

Note how the chosen alias `"tutorial-data"` will be used by subsequent scripts that use these datasets.

### Pretraining

**ODIN** has a pretraining stage where a partial variational autoencoder (PVAE) is trained. To pretrain on the recently generated datasets, run
```bash
uv run scripts/pipeline/pretrain.py --method-name "zannone2019" --dataset cube AFAContext --split 1 2 --launcher <LAUNCHER> --device <DEVICE> --dataset cube AFAContext --dataset-alias tutorial-data --output-alias tutorial-pretrained
```
where `<LAUNCHER>` should be replaced by either `basic` (if you run everything locally in sequence) or the name of the configuration file (without suffix) that you created in `conf/global/hydra/launcher/` if you plan to run everything on a cluster using Slurm.

## Training

The training procedure is very similar to pretraining. The most notable difference is that you now have to provide a set of hard budgets to use for each dataset. If you want to use the budgets [3,5,10] on **cube** but [2,4,8] on **AFAContext**, you would run
```bash
uv run scripts/pipeline/train.py --method-name "zannone2019" --dataset cube AFAContext --budgets "3,5,10" "2,4,8" --split 1 2 --launcher <LAUNCHER> --device <DEVICE> --dataset cube AFAContext --pretrain-alias tutorial-pretrained --output-alias tutorial-trained
```

## Evaluation

One of the main feature of **AFABench** is the consistent evaluation. The same evaluation script is used for all methods. To evaluate your method, either add the name of your method artifact to one of the files in `conf/eval/lists/` or create a new file. If we only wanted to evaluate the method we recently trained, we would create a YAML file with the contents
```yaml
```


```bash
uv run scripts/pipeline/evaluate.py --
```

## Adding New Components

### New AFA Method

1. **Implement the method**:
```python
# src/afa_rl/my_method.py
from common.afa_methods import AFAMethod

class MyAFAMethod(AFAMethod):
    def select(self, features, budget):
        # Your selection logic
        return selected_features
```

2. **Register the method**:
```python
# src/common/registry.py
AFA_METHOD_REGISTRY["my_method"] = MyAFAMethod
```

3. **Add configuration**:
```yaml
# conf/train/my_method/config.yaml
method_type: my_method
# method-specific parameters
```

### New Dataset
1. **Implement dataset class**:
```python
# src/afa_rl/datasets/my_dataset.py
from common.datasets import AFADataset

class MyDataset(AFADataset):
    def generate_data(self):
        # Your data generation logic
        pass
```

2. **Register and configure**:
```python
AFA_DATASET_REGISTRY["my_dataset"] = MyDataset
```

## Citation
If you use this benchmark in your research, please cite,

```bibtex
@inproceedings{
}
```

## License


## Acknowledgments
