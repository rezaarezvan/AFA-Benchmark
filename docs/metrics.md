Each `results.pt` file in the results folder contains a dictionary with the following structure:
- `accuracy_all: Tensor` Average accuracy for each number of acquired features.
- `f1_all: Tensor` Average F1-score for each number of acquired features.
- `feature_mask_history_all: list[list[Tensor]]` Which features where acquired. The outer list represents each sample in the dataset, and the inner list is for each acquired feature for the sample.
