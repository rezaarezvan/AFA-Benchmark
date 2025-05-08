## Required Arguments for Pretraining Scripts

- **`--pretrain_config_path`**
    **Type:** `Path` (required)
    **Description:** Path to the YAML configuration file used for pretraining.

- **`--dataset_type`**
    **Type:** `str` (required)
    **Choices:** Keys from `AFA_DATASET_REGISTRY`
    **Description:** Specifies the type of dataset to use.

- **`--train_dataset_path`**
    **Type:** `Path` (required)
    **Description:** Path to the training dataset.

- **`--val_dataset_path`**
    **Type:** `Path` (required)
    **Description:** Path to the validation dataset.

- **`--pretrained_model_path`**
    **Type:** `Path` (required)
    **Description:** Path to folder to save the pretrained model.

- **`--seed`**
    **Type:** `int` (required)
    **Description:** Random seed for reproducibility.

### Required Arguments for Training Scripts

- **`--pretrain_config_path`**
    **Type:** `Path` (required)
    **Description:** Path to the YAML configuration file used for pretraining.

- **`--train_config_path`**
    **Type:** `Path` (required)
    **Description:** Path to the YAML configuration file for this training.

- **`--dataset_type`**
    **Type:** `str` (required)
    **Choices:** Keys from `AFA_DATASET_REGISTRY`
    **Description:** Specifies the type of dataset to use.

- **`--train_dataset_path`**
    **Type:** `Path` (required)
    **Description:** Path to the training dataset.

- **`--val_dataset_path`**
    **Type:** `Path` (required)
    **Description:** Path to the validation dataset.

- **`--pretrained_model_path`**
    **Type:** `Path` (required)
    **Description:** Path to the folder containing the pretrained model.

- **`--hard_budget`**
    **Type:** `int` (required)
    **Description:** Hard budget value for training.

- **`--seed`**
    **Type:** `int` (required)
    **Description:** Random seed for reproducibility.

- **`--afa_method_path`**
    **Type:** `Path` (required)
    **Description:** Path to the folder where the trained AFA method will be saved.
