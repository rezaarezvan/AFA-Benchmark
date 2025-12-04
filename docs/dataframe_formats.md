## Dataframe produced by evaluation function `eval_afa_method`

- `prev_selections_performed` (list[int]): The selections the method has performed previously.
- `selection_performed` (int): The selection the method performed at this step.
- `builtin_predicted_class` (int|None): The predicted class by the method's built-in classifier, if available.
- `external_predicted_class` (int|None): The predicted class by an external classifier, if provided.
- `true_class` (int): The true class label.

## Dataframe produced by evaluation script `eval_afa_method.py`

- `prev_selections_performed` (list[int]): The selections the method has performed previously.
- `selection_performed` (int): The selection the method performed at this step.
- `builtin_predicted_class` (int|None): The predicted class by the method's built-in classifier, if available.
- `external_predicted_class` (int|None): The predicted class by an external classifier, if provided.
- `true_class` (int): The true class label.
- `eval_seed` (int|null): seed used for evaluating the method (if applicable)
- `eval_hard_budget` (int|null): hard budget used for evaluation (if applicable)

## Dataframe used by plotting script `plot_eval.R`

- `afa_method` (str): name of the AFA method, e.g. "ODIN-MFRL" or "EDDI"
- `classifier` (str): which classifier is used to make the predictions. Values: "builtin" for method's built-in classifier, "external" for external classifier, or "none" if no classifier available.
- `dataset` (str): name of the dataset that the method is evaluated on, e.g. "cube" or "MNIST"
- `selections_performed` (int): how many AFA selections the method has performed for the current sample
- `predicted_class` (int): predicted class using the `classifier`
- `true_class` (int): true class
- `train_seed` (int|null): seed used for training the method (if applicable)
- `eval_seed` (int|null): seed used for evaluating the method (if applicable)
- `hard_budget` (int|null): hard budget used (if applicable)
- `soft_budget_param` (float|null): soft budget parameter used (if applicable)

Note:
- `selections_performed` and `features_observed` are *usually* the same, but not always. For example, we can imagine a scenario where multiple features are unmasked after each AFA action.
