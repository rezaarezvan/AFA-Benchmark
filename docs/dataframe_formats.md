## Expected format for plotting
This is what `plot.R` expects.

- `afa_method` (str): name of the AFA method, e.g. "ODIN-MFRL" or "EDDI"
- `classifier` (str|null): which classifier is used to make the predictions. If null, the AFA method's builtin method is assumed to be used.
- `dataset` (str): name of the dataset that the method is evaluated on, e.g. "cube" or "MNIST"
- `selections_performed` (int): how many AFA selections the method has performed for the current sample
- `features_observed` (int): how many features the method has observed for the current sample
- `predicted_label` (int): predicted class
- `true_label` (int): true class
- `train_seed` (int|null): seed used for training the method (if applicable)
- `eval_seed` (int|null): seed used for evaluating the method (if applicable)
- `train_hard_budget` (int|null): hard budget used during training (if applicable)
- `eval_hard_budget` (int|null): hard budget used during evaluation (if applicable)
- `train_soft_budget_param` (float|null): some parameter provided during training that influences a method's tendency to stop feature collection early (if applicable)
- `eval_soft_budget_param` (float|null): some parameter provided during evaluation that influences a method's tendency to stop feature collection early (if applicable)

Note:
- `selections_performed` and `features_observed` is *usually* the same, but not always. For example, we can imagine a scenario where multiple features are unmasked after each AFA action.
