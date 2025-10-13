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
