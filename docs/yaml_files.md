
## Trained AFAMethods

Each trained AFAMethod is represented by a folder with two files:
- `model.pt` containing the model weights
- `params.yml` containing info about the training

`params.yml` is a YAML file with the following keys:
- `hard_budget: int` Hard budget used during training
- `seed: int` Seed used during training
- `dataset_type: str` Which dataset type was used during training. One of the keys in AFA_DATASET_REGISTRY.
- `train_dataset_path: str` Training dataset used during training
- `val_dataset_path: str` Validation dataset used during training


## Trained AFAClassifiers

Each trained AFAClassifier is represented by a folder with two files:
- `model.pt` containing the model weights
- `params.yml` containing info about the training

`params.yml` is a YAML file with the following keys:
- `seed: int` Seed used during training
- `dataset_type: str` Which dataset type was used during training. One of the keys in AFA_DATASET_REGISTRY.
- `train_dataset_path: str` Training dataset used during training
- `val_dataset_path: str` Validation dataset used during training

## Evaluation results

Combining a trained AFAMethod with a trained AFAClassifier allows us to do evaluations. Each evaluation is represented
by a folder with two files:
- `results.pt` containing the evaluation metrics
- `params.yml` containing info about the evaluation

The `params.yml` file contains some duplicate information from the above files, but it is useful to have it all in one place.

`params.yml` is a YAML file with the following keys:
(Common)
- `dataset_type: str` Which dataset type was used during evaluation. One of the keys in AFA_DATASET_REGISTRY.
(Related to the method)
- `method_hard_budget: int` Hard budget used during training of the AFAMethod
- `method_seed: int` Seed used during training of the AFAMethod
- `method_train_dataset_path: str` Training dataset used during training of the AFAMethod
- `method_val_dataset_path: str` Validation dataset used during training of the AFAMethod
- `method_type: str` Which method was used during training. One of the keys in AFA_METHOD_REGISTRY.
- `method_path: str` Path to the method used.
(Related to the classifier)
- `classifier_seed: int` Seed used during training of the AFAClassifier. Only present if `is_builtin_classifier` is false.
- `classifier_train_dataset_path: str` Training dataset used during training of the AFAClassifier. Only present if `is_builtin_classifier` is false.
- `classifier_val_dataset_path: str` Validation dataset used during training of the AFAClassifier. Only present if `is_builtin_classifier` is false.
- `classifier_type: str` Type of classifier used. One of the keys in AFA_CLASSIFIER_REGISTRY, only present if `is_builtin_classifier` is false.
- `classifier_path: str` Path to the classifier used. Only present if `is_builtin_classifier` is false.
- `is_builtin_classifier = bool` Whether the classifier used was the one built-in in the AFAMethod or not.
(From the evaluation)
- `eval_seed: int` Seed used during evaluation
- `eval_dataset_path: str` Evaluation dataset used during evaluation
- `eval_hard_budget: int` Hard budget used during evaluation. Generally the same as the one used during training, but could be different.
