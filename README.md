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

# Install with uv (recommended)
pip install uv
uv sync

# Or with pip
pip install -e .

# Setup W&B (optional but recommended)
wandb login
```

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
|--------|--------|--------|--------|
| **EDDI** |
| **GDFS** |
| **DIME** |
| **JAFA** |
| **ODIN** |
| **AACO** |
| **JAFA** |
| **PT** |
| **CAE** |

## Datasets

| Dataset | Type | Size (total, # data instances) | # Features | # Classes |
|---------|----------|---------|---------|----------|
| **CUBE** | Synthetic | 1000 | 20 | 8 |
| **AFAContext** | Synthetic | 1000 | 30 | 8 |
| **MNIST** | Real World | 60 000| 784 | 10 |
| **Diabetes** | Real World | 92 063 | 45 | 3 |
| **Physionet** | Real World | 12 000 | 41 | 2

## Full Pipeline Tutorial

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
