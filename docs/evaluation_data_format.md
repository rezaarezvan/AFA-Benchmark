## Soft budget case
After evaluating each method we expect a dataframe with the columns
- `method` (str): name of the method, e.g. "ODIN-MFRL" or "EDDI"
- `train_seed` (int): seed used for training the method
- `cost_param` (float): hyperparameter used during training to control the cost sensitivity. This will have different meanings for different methods.
- `dataset` (str): name of the dataset that the method was trained on and evaluated on (has to be the same)
- `sample` (int): index of the sample in the test set
- `features_chosen` (int): how many features the method chose for this sample before choosing to stop
- `f1_builtin` (float | null): F1 score for this sample using the builtin classifier. `null` if the method has no builtin classifier.
- `f1_external` (float): F1 score for this sample using the external classifier.
- `acc_builtin` (float | null): Accuracy for this sample using the builtin classifier. `null` if the method has no builtin classifier.
- `acc_builtin` (float): Accuracy for this sample using the external classifier.
