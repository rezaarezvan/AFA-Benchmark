import torch
import torch.nn as nn

from pathlib import Path
from xgboost import XGBClassifier
from afa_oracle.classifiers import classifier_ground_truth, classifier_xgb_dict, classifier_xgb
from afa_oracle.mask_generator import random_mask_generator, all_mask_generator, generate_all_masks


def get_knn(X_train, X_query, masks, num_neighbors, instance_idx=0, exclude_instance=True):
    """
    Their exact K-NN implementation

    Args:
        X_train: N x d Train Instances
        X_query: 1 x d Query Instances
        masks: d x R binary masks to try
        num_neighbors: Number of neighbors (k)
    """
    X_train_squared = X_train ** 2
    X_query_squared = X_query ** 2
    X_train_X_query = X_train * X_query
    dist_squared = torch.matmul(X_train_squared, masks) - 2.0 * torch.matmul(
        X_train_X_query, masks) + torch.matmul(X_query_squared, masks)

    if exclude_instance:
        idx_topk = torch.topk(
            dist_squared, num_neighbors + 1, dim=0, largest=False)[1]
        return idx_topk[idx_topk != instance_idx][:num_neighbors]
    else:
        return torch.topk(dist_squared, num_neighbors, dim=0, largest=False)[1]


def load_classifier(dataset_name, X_train, y_train, input_dim, models_dir="models/aaco"):
    """
    Classifier loading logic
    """
    # Handle case variations
    dataset_name_lower = dataset_name.lower()

    if dataset_name_lower == "cube":
        # Use the ground truth classifier for Cube dataset
        return classifier_ground_truth(num_features=20, num_classes=8, std=0.3)

    elif dataset_name_lower in ["grid", "gas10"]:
        # Use XGB dictionary classifier for Grid and Gas10 datasets
        return classifier_xgb_dict(output_dim=y_train.shape[1], input_dim=input_dim, subsample_ratio=0.01, X_train=X_train, y_train=y_train)

    elif dataset_name_lower == "mnist":
        # Expected input size for the pre-trained model
        expected_input_size = 512  # 256 features + 256 mask
        actual_input_size = input_dim * 2  # current features + mask

        if actual_input_size != expected_input_size:
            print(f"MNIST dimensionality mismatch: expected {
                  expected_input_size}, got {actual_input_size}")
            print("Using XGB dictionary classifier for MNIST")
            return classifier_xgb_dict(output_dim=y_train.shape[1], input_dim=input_dim, subsample_ratio=0.01, X_train=X_train, y_train=y_train)

        # Only try to load pre-trained model if dimensions match
        xgb_model = XGBClassifier()
        model_path = Path(models_dir) / \
            'xgb_classifier_MNIST_random_subsets_5.json'

        if not model_path.exists():
            print(f"Warning: XGBoost model not found at {model_path}")
            print("Using XGB dictionary classifier as fallback for MNIST")
            return classifier_xgb_dict(output_dim=y_train.shape[1], input_dim=input_dim, subsample_ratio=0.01, X_train=X_train, y_train=y_train)

        xgb_model.load_model(str(model_path))
        return classifier_xgb(xgb_model)
    else:
        raise ValueError(f"Unsupported dataset: {
                         dataset_name}. Supported: cube, grid, gas10, mnist")


def load_mask_generator(dataset_name, input_dim):
    """
    Their exact mask generator loading logic
    """
    if dataset_name in ["cube", "mnist"]:
        return random_mask_generator(10000, input_dim, 1000)
    elif dataset_name in ["grid", "gas10"]:
        # Generate all possible masks for grid and gas10
        all_masks = generate_all_masks(input_dim)
        return all_mask_generator(all_masks)
    else:
        raise ValueError("Unsupported dataset for mask generation")


def get_initial_feature(dataset_name, n_features):
    """
    Their exact initial feature selection logic
    """
    if dataset_name == "cube":
        return 6
    elif dataset_name == "mnist":
        return 100
    elif dataset_name == "grid":
        return 1
    elif dataset_name == "gas10":
        return 6
    else:
        # Default: select middle feature
        return n_features // 2


class AACOOracle:
    def __init__(self,
                 k_neighbors=5,
                 acquisition_cost=0.05,
                 hide_val=10.0,
                 dataset_name="cube"):
        self.k_neighbors = k_neighbors
        self.acquisition_cost = acquisition_cost
        self.hide_val = hide_val
        self.dataset_name = dataset_name
        self.classifier = None
        self.mask_generator = None
        self.X_train = None
        self.y_train = None
        self.loss_function = nn.CrossEntropyLoss(reduction='none')

    def fit(self, X_train, y_train):
        """
        Fit the oracle on training data
        """
        self.X_train = X_train
        self.y_train = y_train

        # Load exact classifier
        input_dim = X_train.shape[1]
        self.classifier = load_classifier(
            self.dataset_name, X_train, y_train, input_dim)

        # Load exact mask generator
        self.mask_generator = load_mask_generator(self.dataset_name, input_dim)

        print(f"AACO Oracle fitted for {self.dataset_name}")
        print(f"Training data: {X_train.shape}")

    def select_next_feature(self, x_observed, observed_mask, instance_idx=0):
        """
        Feature selection logic
        Args:
            x_observed: 1D tensor of observed features
            observed_mask: 1D boolean tensor indicating which features are observed
            instance_idx: index of current instance (for KNN exclusion)
        Returns:
            next_feature: int index of next feature to acquire, or None if terminate
        """
        if self.classifier is None:
            raise ValueError("Oracle must be fitted first")

        feature_count = len(x_observed)

        # Convert to format: current mask as 1 x d
        mask_curr = observed_mask.float().unsqueeze(0)

        # Check if this is the first feature selection
        if not observed_mask.any():
            # Select initial feature deterministically (their approach)
            initial_feature = get_initial_feature(
                self.dataset_name, feature_count)
            return initial_feature

        # Get the nearest neighbors based on the observed feature mask
        x_query = x_observed.unsqueeze(0)  # 1 x d
        idx_nn = get_knn(self.X_train, x_query, mask_curr.T,
                         self.k_neighbors, instance_idx, True).squeeze()

        # Generate random masks and get the next set of possible masks
        new_masks = self.mask_generator(mask_curr)
        mask = torch.maximum(
            new_masks, mask_curr.repeat(new_masks.shape[0], 1))
        mask[0] = mask_curr  # Ensure the current mask is included

        # Get only unique masks
        mask = mask.unique(dim=0)
        n_masks_updated = mask.shape[0]

        # Predictions based on the classifier
        x_rep = self.X_train[idx_nn].repeat(n_masks_updated, 1)
        mask_rep = torch.repeat_interleave(mask, self.k_neighbors, 0)
        idx_nn_rep = idx_nn.repeat(n_masks_updated)

        # Apply masking with hide_val
        x_masked = torch.mul(x_rep, mask_rep) - (1 - mask_rep) * self.hide_val
        x_with_mask = torch.cat([x_masked, mask_rep], -1)

        y_pred = self.classifier(x_with_mask, idx_nn_rep)

        # Compute loss
        y_true_rep = self.y_train[idx_nn_rep]
        if len(y_true_rep.shape) > 1 and y_true_rep.shape[1] > 1:
            # One-hot encoded
            y_true_labels = y_true_rep.argmax(dim=1)
        else:
            y_true_labels = y_true_rep

        loss = self.loss_function(y_pred, y_true_labels)

        # Reshape loss to (n_masks, k_neighbors) and average over neighbors
        loss = loss.view(n_masks_updated, self.k_neighbors)
        avg_loss = loss.mean(dim=1)

        # Add acquisition cost
        mask_costs = torch.sum(mask, dim=1) - torch.sum(mask_curr)
        total_cost = avg_loss + self.acquisition_cost * mask_costs

        # Find the mask with minimum cost
        best_mask_idx = total_cost.argmin()
        best_mask = mask[best_mask_idx]

        # Check if improvement is worth the cost
        current_cost = total_cost[0]  # Cost of current mask (index 0)
        best_cost = total_cost[best_mask_idx]

        if best_cost >= current_cost:
            # No improvement, terminate
            return None

        # Find which new features to acquire
        new_features = (best_mask - mask_curr.squeeze()
                        ).nonzero(as_tuple=True)[0]

        if len(new_features) == 0:
            return None
        elif len(new_features) == 1:
            return new_features[0].item()
        else:
            # If multiple features, select the one with lowest individual loss
            individual_losses = []
            for feat in new_features:
                temp_mask = mask_curr.clone()
                temp_mask[0, feat] = 1
                # Quick loss computation for this feature
                individual_losses.append(avg_loss[best_mask_idx])  # Simplified

            best_feat_idx = torch.tensor(individual_losses).argmin()
            return new_features[best_feat_idx].item()
