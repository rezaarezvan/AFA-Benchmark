## Soft budget case
After evaluating each method we expect a dataframe with the columns
- `Method` (str): name of the method, e.g. "ODIN-MFRL" or "EDDI"
- `Training seed` (int): seed used for training the method
- `Cost parameter` (float): hyperparameter used during training to control the cost sensitivity. This will have different meanings for different methods.
- `Dataset` (str): name of the dataset that the method was trained on and evaluated on (has to be the same)
- `Sample` (int): index of the sample in the test set
- `Features chosen` (int): how many features the method chose for this sample before choosing to stop
- `F1 (builtin)` (float | null): F1 score for this sample using the builtin classifier. `null` if the method has no builtin classifier.
- `F1 (external)` (float): F1 score for this sample using the external classifier.
- `Accuracy (builtin)` (float | null): Accuracy for this sample using the builtin classifier. `null` if the method has no builtin classifier.
- `Accuracy (external)` (float): Accuracy for this sample using the external classifier.
