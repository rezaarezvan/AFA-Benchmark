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

## Terminology

- AFA method / method: An algorithm that sequentially selects features to acquire, based on the current state of acquired features and the data. Is usually also able to make label predictions in each step.
- Model: A neural network.
- Dataset/method/classifier type: "Type" refers to strings that can be passed to the functions in the `registry` module.

## Artifact structure

### Datasets

Each dataset artifact is expected to contain a `dataset_type` key in its metadata. Passing this string to the `get_afa_dataset_class` function should return the correct dataset class. The `load` method of the dataset class is then called on the files "train.pt", "val.pt" and "test.pt" in the artifact. See `load_dataset_artifact` for more info.

### Trained methods (AFA methods)

Each trained method should have the following keys in its metadata:
- `afa_method_class` (`str`): A string that can be passed to the `get_afa_method_class` function to return the correct AFA method class. The `load` method of this class is then called with the contents of the artifact. See `load_trained_method_artifact` for more info.
- `budget`` (`int|None`): Which budget this method was trained with. For some methods, this is not applicable, in which case it should be set to `None`.
- `dataset_artifact_name` (`str`): Which artifact this method was trained on. This is used to load the same dataset artifact with `load_dataset_artifact` during evaluation.
- `dataset_type` (`str`): The dataset type of `dataset_artifact_name`. Superfluous but convenient.
- `method_type` (`str`): The method type of the trained afa method.
- `seed` (`int`): The random seed used for training this method.

Note that `afa_method_class` can be the same for several different `method_type`s. This is the case for the RL methods which have a common AFA method class `RLAFAMethod` but have different `method_type`s like `"shim2018"` and `"zannone2019"` in order to distinguish them during evaluation.

### Evaluation results

The `eval_afa_method.py` script generates artifacts with the following keys in their metadata:
- `budget` (`int`): The budget used for evaluation. If the trained method artifact had a budget set, the same budget is used here. If the trained method artifact had no budget set, the budget is equal to the number of features in the dataset.
- `classifier_type` (`str`): The classifier type used for evaluation. Either a "normal" classifier type that can be passed to `get_afa_classifier_class`, or `"builtin"`, in which case the built-in classifier of the trained method was used.
- `dataset_type` (`str`): The dataset type of the dataset used for evaluation. Always the same as the `dataset_type` of the trained method artifact.
- `method_type` (`str`): The method type of the trained method artifact.
- `seed` (`int`): The random seed used for evaluation.

## Hydra

All scripts in this repository use [Hydra](https://hydra.cc/) for configuration management, mainly to simplify batching of experiments. [Structured configs](https://hydra.cc/docs/tutorials/structured_config/intro/) are defined in `src/common/config_classes.py` and the configurations themselves are defined in the `conf` directory.

## Full pipeline example using shim2018

TODO
