import os
import torch
import wandb
import joblib
import logging
import numpy as np
import wandb.errors
import torch.nn as nn

from scipy.stats import norm
from xgboost import XGBClassifier
from tempfile import NamedTemporaryFile

log = logging.getLogger(__name__)


class NaiveBayes(nn.Module):
    """
    Their exact Naive Bayes implementation for Cube dataset
    """

    def __init__(self, num_features=20, num_classes=8, std=0.3):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.std = std

    def forward(self, x):
        """
        Args:
            x: data tensor of shape (N, d)
        """
        try:
            mask = x[:, self.num_features:]
            x = x[:, :self.num_features]
        except IndexError:
            log.error(
                "Classifier expects masking information to be concatenated with each feature vector.")

        device = x.device
        y_classes = list(range(self.num_classes))
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
    def __init__(self, output_dim, input_dim, subsample_ratio, X_train, y_train,
                 device=None, dataset_name="unknown", save_batch_size=100):
        """
        Input:
            output_dim: Dimension of the outcome y
            input_dim: Dimension of the input features (X)
            subsample_ratio: Fraction of training points for each boosting iteration
            device: torch.device to use
            dataset_name: Name of dataset for artifact naming
            save_batch_size: Number of classifiers to train before saving to wandb
        """
        self.xgb_model_dict = {}
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.subsample_ratio = subsample_ratio
        self.dataset_name = dataset_name
        self.save_batch_size = save_batch_size
        self.classifiers_trained_since_save = 0

        # Use provided device or infer from data
        if device is not None:
            self.device = device
        else:
            self.device = X_train.device if hasattr(
                X_train, 'device') else torch.device('cpu')

        # Convert to numpy for XGBoost
        if torch.is_tensor(X_train):
            self.X_train = X_train.cpu().numpy()
        else:
            self.X_train = X_train

        if torch.is_tensor(y_train):
            if len(y_train.shape) > 1 and y_train.shape[1] > 1:
                self.y_train = y_train.argmax(dim=1).cpu().numpy()
            else:
                self.y_train = y_train.cpu().numpy()
        else:
            self.y_train = y_train

        # XGBoost parameters - use consistent params with single XGB
        self.xgb_params = {
            'n_estimators': 100,  # Smaller for dict since we train many
            'max_depth': 6,       # Smaller depth for dict
            'random_state': 29,
            'n_jobs': 4,          # Fewer jobs since we train many in parallel
            'tree_method': 'hist',
        }

        log.info(f"XGBoost dictionary initialized for {dataset_name}")
        log.info(f"Periodic saving enabled: every {
                 save_batch_size} classifiers")

        # Try to load existing classifiers
        self._load_classifiers_from_wandb()

    def _get_dict_artifact_name(self):
        """Generate artifact name for XGB dictionary"""
        return f"xgb_dict_{self.dataset_name}_{self.input_dim}d_{self.output_dim}c"

    def _load_classifiers_from_wandb(self):
        """Try to load existing XGB dictionary from wandb"""
        if not wandb.run:
            log.warning("No active wandb run, skipping classifier load")
            return

        artifact_name = self._get_dict_artifact_name()

        try:
            artifact = wandb.use_artifact(f"{artifact_name}:latest")
            artifact_dir = artifact.download()

            dict_path = os.path.join(artifact_dir, "xgb_dict.joblib")
            if os.path.exists(dict_path):
                self.xgb_model_dict = joblib.load(dict_path)
                log.info(f"Loaded {len(self.xgb_model_dict)} XGB classifiers from artifact: {
                         artifact_name}")

        except wandb.errors.CommError:
            log.warning("No existing XGB dictionary found for "
                        f"{self.dataset_name}, starting fresh.")
        except Exception as e:
            log.error(f"Failed to load XGB dictionary from wandb: {e}")

    def _save_classifiers_to_wandb(self, force_save=False):
        """Save XGB dictionary to wandb artifact"""
        if not wandb.run:
            return
        if not force_save and self.classifiers_trained_since_save < self.save_batch_size:
            return

        artifact_name = self._get_dict_artifact_name()

        try:
            with NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_file:
                joblib.dump(self.xgb_model_dict, tmp_file.name)

                artifact = wandb.Artifact(
                    name=artifact_name,
                    type="xgb_dict_classifier",
                    metadata={
                        "dataset_name": self.dataset_name,
                        "input_dim": self.input_dim,
                        "output_dim": self.output_dim,
                        "num_classifiers": len(self.xgb_model_dict),
                        "subsample_ratio": self.subsample_ratio,
                    }
                )

                artifact.add_file(tmp_file.name, name="xgb_dict.joblib")
                wandb.log_artifact(artifact)

                os.unlink(tmp_file.name)

            log.info(f"Saved {len(self.xgb_model_dict)
                              } XGB classifiers to artifact: {artifact_name}")
            self.classifiers_trained_since_save = 0

        except Exception as e:
            log.error(f"Failed to save XGB dictionary to wandb: {e}")

    def __call__(self, X, idx):
        device = X.device if hasattr(X, 'device') else torch.device('cpu')
        n = X.shape[0]
        probs = torch.zeros((n, self.output_dim), device=device)

        # First pass: train any missing classifiers
        masks_to_train = []
        for i in range(n):
            mask_i = X[i][self.input_dim:].cpu()
            mask_i_string = ''.join(map(str, mask_i.long().tolist()))

            if mask_i_string not in self.xgb_model_dict:
                masks_to_train.append((i, mask_i, mask_i_string))

        # Train missing classifiers
        for i, mask_i, mask_i_string in masks_to_train:
            try:
                nonzero_i = mask_i.nonzero().squeeze()

                if nonzero_i.dim() == 0:
                    nonzero_i = nonzero_i.unsqueeze(0)
                if len(nonzero_i) == 0:
                    self.xgb_model_dict[mask_i_string] = None
                    continue

                # Train classifier for this mask
                X_train_subset = self.X_train[:, nonzero_i].reshape(
                    self.X_train.shape[0], -1)
                n_samples = min(len(X_train_subset), int(
                    len(X_train_subset) * self.subsample_ratio))
                n_samples = max(n_samples, 100)  # Minimum samples

                idx_sample = np.random.choice(
                    X_train_subset.shape[0], n_samples, replace=False)

                self.xgb_model_dict[mask_i_string] = XGBClassifier(
                    **self.xgb_params)
                self.xgb_model_dict[mask_i_string].fit(
                    X_train_subset[idx_sample], self.y_train[idx_sample]
                )
                self.classifiers_trained_since_save += 1

                # Periodic saving
                if self.classifiers_trained_since_save >= self.save_batch_size:
                    log.info(f"Saving progress: {
                             len(self.xgb_model_dict)} classifiers trained so far...")
                    self._save_classifiers_to_wandb()

            except Exception as e:
                log.error(f"Failed to train XGBoost classifier for mask {
                          mask_i_string}: {e}")
                self.xgb_model_dict[mask_i_string] = None

        # Final save if there are unsaved classifiers
        if self.classifiers_trained_since_save > 0:
            self._save_classifiers_to_wandb(force_save=True)

        # Second pass: make predictions
        for i in range(n):
            mask_i = X[i][self.input_dim:].cpu()
            nonzero_i = mask_i.nonzero().squeeze()

            if nonzero_i.dim() == 0:
                nonzero_i = nonzero_i.unsqueeze(0)

            if len(nonzero_i) == 0:
                probs[i] = torch.ones(
                    self.output_dim, device=device) / self.output_dim
                continue

            mask_i_string = ''.join(map(str, mask_i.long().tolist()))

            if self.xgb_model_dict[mask_i_string] is not None:
                try:
                    X_query = X[i, nonzero_i].cpu().numpy().reshape(1, -1)
                    pred_probs = self.xgb_model_dict[mask_i_string].predict_proba(
                        X_query)
                    probs[i] = torch.from_numpy(pred_probs[0]).to(device)
                except Exception as e:
                    log.error(f"Prediction failed for mask {
                              mask_i_string}: {e}")
                    probs[i] = torch.ones(
                        self.output_dim, device=device) / self.output_dim
            else:
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
    Single XGBoost classifier that expects [features + mask] as concatenated input
    Used for MNIST and other high-dimensional datasets
    """

    def __init__(self, xgb_model):
        self.xgb_model = xgb_model

    def predict_logits(self, X):
        if torch.is_tensor(X):
            X = X.cpu().numpy()
        return torch.tensor(
            self.xgb_model.predict(X, output_margin=True),
            dtype=torch.float32,
        )

    def __call__(self, X, idx):
        device = X.device if hasattr(X, 'device') else torch.device('cpu')

        # Convert to numpy for XGBoost prediction
        if torch.is_tensor(X):
            X_np = X.detach().cpu().numpy().astype(np.float32)
        else:
            X_np = X.astype(np.float32)

        # XGBoost expects 2D input: [batch_size, features]
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)

        pred_probs = self.xgb_model.predict_proba(X_np)

        # Convert back to torch tensor on correct device
        result = torch.tensor(pred_probs, device=device, dtype=torch.float32)

        # Handle single sample case
        if result.dim() == 1:
            result = result.unsqueeze(0)

        return result
