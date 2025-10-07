# Evaluation data format

Note that the hard and soft budget cases have the same column names, but `Features chosen` has a different meaning.

During plotting, we can join the dataset dataframe with the results dataframe on `Dataset` and `Sample` to get the true labels.

## Dataset
- `Dataset` (str): name of a dataset
- `Sample` (int): index of the sample
- `True label` (int): true label for this sample

## Hard budget case
After evaluating each method we expect a dataframe with the columns
- `Method` (str): name of the method, e.g. "ODIN-MFRL" or "EDDI"
- `Training seed` (int): seed used for training the method
- `Dataset` (str): name of the dataset that the method was trained on and evaluated on (has to be the same)
- `Sample` (int): index of the sample
- `Features chosen` (int): how many features the method has chosen so far for this sample
- `Predicted label (builtin)` (int | null): Predicted label for this sample using the builtin classifier. `null` if the method has no builtin classifier.
- `Predicted label (external)` (float): Predicted label for this sample using the external classifier.

## Soft budget case
After evaluating each method we expect a dataframe with the columns
- `Method` (str): name of the method, e.g. "ODIN-MFRL" or "EDDI"
- `Training seed` (int): seed used for training the method
- `Cost parameter` (float): hyperparameter used during training to control the cost sensitivity. This will have different meanings for different methods.
- `Dataset` (str): name of the dataset that the method was trained on and evaluated on (has to be the same)
- `Sample` (int): index of the sample
- `Features chosen` (int): how many features the method chose for this sample before choosing to stop
- `Predicted label (builtin)` (int | null): Predicted label for this sample using the builtin classifier. `null` if the method has no builtin classifier.
- `Predicted label (external)` (float): Predicted label for this sample using the external classifier.
