# Evaluation data format
Note that the hard and soft budget cases have the same column names, but `features_chosen` has a different meaning.

## Hard budget case
After evaluating each method we expect a dataframe with the columns
- `method` (str): name of the method, e.g. "ODIN-MFRL" or "EDDI"
- `training_seed` (int): seed used for training the method
- `dataset` (str): name of the dataset that the method was trained on and evaluated on (has to be the same)
- `features_chosen` (int): how many features the method has chosen so far for this sample
- `predicted_label_builtin` (int | null): Predicted label for this sample using the builtin classifier. `null` if the method has no builtin classifier.
- `predicted_label_external` (float): Predicted label for this sample using the external classifier.
- `true_label` (float): True label for this sample

## Soft budget case
After evaluating each method we expect a dataframe with the columns
- `method` (str): name of the method, e.g. "ODIN-MFRL" or "EDDI"
- `training_seed` (int): seed used for training the method
- `cost_parameter` (float): hyperparameter used during training to control the cost sensitivity. This will have different meanings for different methods.
- `dataset` (str): name of the dataset that the method was trained on and evaluated on (has to be the same)
- `features_chosen` (int): how many features the method chose for this sample before choosing to stop
- `acquisition_cost` (float): total acquisition cost for this sample
- `predicted_label_builtin` (int | null): Predicted label for this sample using the builtin classifier. `null` if the method has no builtin classifier.
- `predicted_label_external` (float): Predicted label for this sample using the external classifier.
- `true_label` (float): True label for this sample

## `plot.R`
Note:
- `actions_taken` and `features_observed` is *usually* the same, but not always. For example, we can imagine a scenario where multiple features are unmasked after each AFA action.

- `afa_method` (str): name of the AFA method, e.g. "ODIN-MFRL" or "EDDI"
- `classifier` (str|null): which classifier is used to make the predictions. If null, the AFA method's builtin method is assumed to be used.
- `dataset` (str): name of the dataset that the method is evaluated on, e.g. "cube" or "MNIST"
- `actions_taken` (int): how many AFA actions the method has taken for the current sample
- `features_observed` (int): how many features the method has observed for the current sample
- `predicted_label` (int): predicted class
- `true_label` (int): true class
- `train_seed` (int|null): seed used for training the method (if applicable)
- `eval_seed` (int|null): seed used for evaluating the method (if applicable)
- `train_hard_budget` (int|null): hard budget used during training (if applicable)
- `eval_hard_budget` (int|null): hard budget used during evaluation (if applicable)
- `train_soft_budget_param` (float|null): some parameter provided during training that influences a method's tendency to stop feature collection early (if applicable)
- `eval_soft_budget_param` (float|null): some parameter provided during evaluation that influences a method's tendency to stop feature collection early (if applicable)
