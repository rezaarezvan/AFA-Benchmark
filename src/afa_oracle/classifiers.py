import torch
import pickle
import wandb
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from pathlib import Path
from scipy.stats import norm
from xgboost import XGBClassifier


class NaiveBayes(nn.Module):
    """
    Their exact Naive Bayes implementation for Cube dataset
    """

    def __init__(self, num_features=20, num_classes=8, std=0.3):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.std = std

    def forward(self, x, mask):
        """
        Args:
            x: data tensor of shape (N, d)
            mask: mask tensor of shape (N, d)
        """
        device = x.device
        y_classes = list(range(self.num_classes))

        # Initialize output_probs on the correct device
        output_probs = torch.zeros((len(x), self.num_classes), device=device)

        for y_val in y_classes:
            # PDF values for each feature in x conditioned on the given label y_val

            # Default to PDF for U[0,1) - ensure tensors are on correct device
            p_x_y = torch.where((x >= 0) & (x < 1),
                                torch.ones(x.shape, device=device),
                                torch.zeros(x.shape, device=device))

            # Use normal distribution PDFs for appropriate features given y_val
            p_x_y[:, y_val:y_val+3] = torch.transpose(
                torch.Tensor(np.array([norm.pdf(x[:, y_val].cpu(), y_val % 2, self.std),
                                       norm.pdf(x[:, y_val+1].cpu(),
                                                (y_val // 2) % 2, self.std),
                                       norm.pdf(x[:, y_val+2].cpu(), (y_val // 4) % 2, self.std)])).to(device), 0, 1)

            # Compute joint probability over masked features
            p_xo_y = torch.prod(torch.where(torch.gt(mask, 0), p_x_y,
                                            torch.tensor(1.0, device=device)), dim=1)

            p_y = torch.tensor(1.0 / self.num_classes, device=device)

            output_probs[:, y_val] = p_xo_y * p_y

        # Normalize properly and avoid division by zero
        normalizer = torch.sum(output_probs, dim=1, keepdim=True)
        normalizer = torch.clamp(normalizer, min=1e-8)
        return output_probs / normalizer

    def predict(self, x):
        return self.forward(x)


class classifier_xgb_dict():
    """
    XGBoost dictionary classifier with proper GPU support and progress indicators
    """

    def __init__(self, output_dim, input_dim, subsample_ratio, X_train, y_train, device=None, dataset_name="mnist"):
        """
        Input:
        output_dim: Dimension of the outcome y
        input_dim: Dimension of the input features (X)
        subsample_ratio: Fraction of training points for each boosting iteration
        device: torch.device to use ('cpu' or 'cuda')
        dataset_name: Name of dataset for caching
        """
        self.xgb_model_dict = {}
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.subsample_ratio = subsample_ratio
        self.dataset_name = dataset_name

        # Use provided device or infer from data
        if device is not None:
            self.device = device
        else:
            self.device = X_train.device if hasattr(
                X_train, 'device') else torch.device('cpu')

        # Convert to numpy for XGBoost, but keep track of device
        if torch.is_tensor(X_train):
            self.X_train = X_train.cpu().numpy()
        else:
            self.X_train = X_train

        if torch.is_tensor(y_train):
            if len(y_train.shape) > 1 and y_train.shape[1] > 1:
                # One-hot encoded
                self.y_train = y_train.argmax(dim=1).cpu().numpy()
            else:
                self.y_train = y_train.cpu().numpy()
        else:
            self.y_train = y_train

        # XGBoost parameters - always use CPU to avoid device mismatch warnings
        # since we work with numpy arrays anyway
        self.xgb_params = {
            'n_estimators': 250,
            'max_depth': 5,
            'random_state': 29,
            'n_jobs': 8,
            'tree_method': 'hist',
        }

        print(f"XGBoost classifier initialized with device: CPU (to avoid GPU/CPU data mismatch)")

        # Try to load existing classifiers
        self._load_classifiers_from_wandb()

    def _get_classifier_artifact_name(self):
        """Generate artifact name for classifiers cache"""
        return f"xgb_classifiers_{self.dataset_name.lower()}_{self.input_dim}_{self.output_dim}"

    def _load_classifiers_from_wandb(self):
        """Try to load existing classifiers from wandb artifacts"""
        try:
            artifact_name = self._get_classifier_artifact_name()
            artifact = wandb.use_artifact(
                f"{artifact_name}:latest", type="classifier_cache")
            artifact_dir = Path(artifact.download())

            classifiers_path = artifact_dir / "classifiers.pkl"
            if classifiers_path.exists():
                with open(classifiers_path, 'rb') as f:
                    self.xgb_model_dict = pickle.load(f)
                print(f"Loaded {len(self.xgb_model_dict)
                                } cached XGBoost classifiers from wandb")
                return True
        except Exception as e:
            print(f"Could not load cached classifiers: {e}")
        return False

    def _save_classifiers_to_wandb(self):
        """Save trained classifiers to wandb artifacts"""
        try:
            import tempfile

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                classifiers_path = tmp_path / "classifiers.pkl"

                # Save the classifiers dict
                with open(classifiers_path, 'wb') as f:
                    pickle.dump(self.xgb_model_dict, f)

                # Create wandb artifact
                artifact_name = self._get_classifier_artifact_name()
                artifact = wandb.Artifact(
                    name=artifact_name,
                    type="classifier_cache",
                    metadata={
                        "dataset": self.dataset_name,
                        "input_dim": self.input_dim,
                        "output_dim": self.output_dim,
                        "n_classifiers": len(self.xgb_model_dict),
                        "subsample_ratio": self.subsample_ratio,
                    }
                )
                artifact.add_file(str(classifiers_path))
                wandb.log_artifact(artifact)
                print(f"Saved {len(self.xgb_model_dict)
                               } XGBoost classifiers to wandb")

        except Exception as e:
            print(f"Could not save classifiers to wandb: {e}")

    def __call__(self, X, idx):
        n = X.shape[0]
        device = X.device if hasattr(X, 'device') else torch.device('cpu')
        probs = torch.zeros((n, self.output_dim), device=device)

        # Check if we're using GPU XGBoost
        using_gpu_xgb = self.device.type == 'cuda' and torch.cuda.is_available()

        # Track progress for classifier building
        masks_to_build = []
        cached_masks = 0

        # First pass: identify which masks need to be built
        for i in range(n):
            mask_i = X[i][self.input_dim:].cpu()
            nonzero_i = mask_i.nonzero().squeeze()

            # Handle edge case: ensure nonzero_i is always an array
            if nonzero_i.dim() == 0:  # scalar case (single feature)
                nonzero_i = nonzero_i.unsqueeze(0)

            # Check if no features are selected
            if len(nonzero_i) == 0:
                # No features selected - return uniform probabilities
                dummy_probs = torch.ones(
                    self.output_dim, device=device) / self.output_dim
                probs[i] = dummy_probs
                continue

            mask_i_string = ''.join(map(str, mask_i.long().tolist()))

            # Check if the mask is already in the dictionary
            if mask_i_string not in self.xgb_model_dict:
                masks_to_build.append((i, mask_i_string, nonzero_i))
            else:
                cached_masks += 1

        # Build new classifiers with progress bar
        new_classifiers_added = False
        if masks_to_build:
            print(f"Building {len(masks_to_build)} new XGBoost classifiers (using {
                  cached_masks} cached)...")

            for idx, (i, mask_i_string, nonzero_i) in enumerate(tqdm(masks_to_build, desc="Training classifiers")):
                try:
                    self.xgb_model_dict[mask_i_string] = XGBClassifier(
                        **self.xgb_params)

                    # Extract features for selected indices
                    X_train_subset = self.X_train[:, nonzero_i.cpu().numpy()]

                    # Ensure we have the right shape
                    if X_train_subset.ndim == 1:
                        X_train_subset = X_train_subset.reshape(-1, 1)

                    # Subsample training data
                    n_samples = max(
                        1, int(X_train_subset.shape[0] * self.subsample_ratio))
                    idx_sample = np.random.choice(
                        X_train_subset.shape[0], n_samples, replace=False)

                    self.xgb_model_dict[mask_i_string].fit(
                        X_train_subset[idx_sample], self.y_train[idx_sample])
                    new_classifiers_added = True

                except Exception as e:
                    print(f"Warning: Failed to train XGBoost classifier for mask {
                          mask_i_string}: {e}")
                    # Fallback: create a dummy classifier that returns uniform probabilities
                    self.xgb_model_dict[mask_i_string] = None

        # Save new classifiers to wandb if any were added
        if new_classifiers_added:
            self._save_classifiers_to_wandb()

        # Second pass: make predictions
        for i in range(n):
            mask_i = X[i][self.input_dim:].cpu()
            nonzero_i = mask_i.nonzero().squeeze()

            # Handle edge case: ensure nonzero_i is always an array
            if nonzero_i.dim() == 0:  # scalar case (single feature)
                nonzero_i = nonzero_i.unsqueeze(0)

            # Check if no features are selected
            if len(nonzero_i) == 0:
                # No features selected - return uniform probabilities
                dummy_probs = torch.ones(
                    self.output_dim, device=device) / self.output_dim
                probs[i] = dummy_probs
                continue

            mask_i_string = ''.join(map(str, mask_i.long().tolist()))

            # Make prediction
            if self.xgb_model_dict[mask_i_string] is not None:
                try:
                    # Extract query data and handle device properly
                    X_query = X[i, nonzero_i]

                    # For GPU XGBoost, try to keep data on GPU if possible
                    if using_gpu_xgb and hasattr(X_query, 'cuda'):
                        # Keep on GPU and convert to numpy - XGBoost will handle GPU transfer
                        X_query_np = X_query.detach().cpu().numpy().reshape(1, -1)
                    else:
                        # Standard CPU conversion
                        X_query_np = X_query.cpu().numpy().reshape(1, -1)

                    pred_probs = self.xgb_model_dict[mask_i_string].predict_proba(
                        X_query_np)
                    probs[i] = torch.from_numpy(pred_probs[0]).to(device)
                except Exception as e:
                    print(f"Warning: Prediction failed for mask {
                          mask_i_string}: {e}")
                    # Fallback to uniform probabilities
                    probs[i] = torch.ones(
                        self.output_dim, device=device) / self.output_dim
            else:
                # Use fallback uniform probabilities
                probs[i] = torch.ones(
                    self.output_dim, device=device) / self.output_dim

        return probs


class classifier_ground_truth():
    """
    Wrapper for ground truth NaiveBayes classifier
    """

    def __init__(self, num_features=20, num_classes=8, std=0.3):
        self.gt_classifier = NaiveBayes(
            num_features=num_features, num_classes=num_classes, std=std)

    def __call__(self, X, idx):
        device = X.device if hasattr(X, 'device') else torch.device('cpu')
        self.gt_classifier = self.gt_classifier.to(device)
        return self.gt_classifier.predict(X)


class classifier_xgb():
    """
    Wrapper for pre-trained XGBoost models with proper device handling
    """

    def __init__(self, xgb_model):
        self.xgb_model = xgb_model

    def __call__(self, X, idx):
        device = X.device if hasattr(X, 'device') else torch.device('cpu')

        # Convert to numpy for XGBoost prediction
        if torch.is_tensor(X):
            X_np = X.cpu().numpy()
        else:
            X_np = X

        try:
            pred_probs = self.xgb_model.predict_proba(X_np)
            return torch.tensor(pred_probs, device=device)
        except Exception as e:
            print(f"Warning: XGBoost prediction failed: {e}")
            # Return uniform probabilities as fallback
            n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
            n_classes = getattr(self.xgb_model, 'n_classes_', 2)
            uniform_probs = torch.ones(
                (n_samples, n_classes), device=device) / n_classes
            return uniform_probs
