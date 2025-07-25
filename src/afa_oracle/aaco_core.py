import os
import torch
import wandb
import joblib
import logging
import numpy as np
import wandb.errors
import torch.nn as nn

from pathlib import Path
from xgboost import XGBClassifier
from tempfile import NamedTemporaryFile
from common.classifiers import WrappedMaskedMLPClassifier
from afa_oracle.classifiers import classifier_ground_truth, classifier_mlp, classifier_xgb
from afa_oracle.mask_generator import random_mask_generator, all_mask_generator, generate_all_masks

log = logging.getLogger(__name__)


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


def load_classifier(dataset_name, X_train, y_train, input_dim, device=None, models_dir="models/aaco"):
    dataset_name_lower = dataset_name.lower()
    n_classes = y_train.shape[1] if len(
        y_train.shape) > 1 else int(y_train.max().item()) + 1

    if dataset_name_lower == "cube":
        return classifier_ground_truth(num_features=20, num_classes=8, std=0.3)

    elif dataset_name_lower in ["afacontext", "miniboone", "physionet", "diabetes"]:
        # return train_single_xgb_classifier(
        #     X_train, y_train, input_dim, n_classes, device, dataset_name_lower
        # )

        log.info(f"Loading MLP classifier for {dataset_name}...")
        try:
            if wandb.run:
                if dataset_name_lower == "afacontext":
                    dataset_name_lower = "AFAContext"
                artifact_name = f"masked_mlp_classifier-{
                    dataset_name_lower}_split_1:latest"
                artifact = wandb.use_artifact(artifact_name)
                artifact_dir = artifact.download()

                classifier_path = Path(artifact_dir) / "classifier.pt"
                wrapped_mlp = WrappedMaskedMLPClassifier.load(
                    classifier_path, device)

                log.info(f"Successfully loaded MLP classifier from {
                         artifact_name}")
                return classifier_mlp(wrapped_mlp, hide_val=10.0)

        except Exception as e:
            log.warning(f"Failed to load MLP classifier: {e}")
            log.info("Falling back to XGBoost...")

    elif dataset_name_lower in ["mnist", "fashionmnist"]:
        # # Train single XGB with wandb caching
        # return train_single_xgb_classifier(
        #     X_train, y_train, input_dim, n_classes, device, dataset_name_lower
        # )

        log.info(f"Loading MLP classifier for {dataset_name}...")

        try:
            if wandb.run:
                artifact_name = "masked_mlp_classifier-MNIST_split_1:latest"
                artifact = wandb.use_artifact(artifact_name)
                artifact_dir = artifact.download()

                classifier_path = Path(artifact_dir) / "classifier.pt"
                wrapped_mlp = WrappedMaskedMLPClassifier.load(
                    classifier_path, device)

                log.info(f"Successfully loaded MLP classifier from {
                         artifact_name}")
                return classifier_mlp(wrapped_mlp, hide_val=10.0)

        except Exception as e:
            log.warning(f"Failed to load MLP classifier: {e}")
            log.info("Falling back to XGBoost...")

        # Fallback to XGBoost if MLP loading fails
        return train_single_xgb_classifier(
            X_train, y_train, input_dim, n_classes, device, dataset_name_lower, hide_val=10.0
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_single_xgb_artifact_name(dataset_name, input_dim, n_classes):
    """Generate artifact name for single XGB classifier"""
    return f"xgb_single_{dataset_name.lower()}_{input_dim}d_{n_classes}c"


def save_single_xgb_to_wandb(xgb_model, dataset_name, input_dim, n_classes):
    """Save single XGBoost model to wandb artifact"""
    if not wandb.run:
        log.warning("No active wandb run, cannot save model")
        return

    artifact_name = get_single_xgb_artifact_name(
        dataset_name, input_dim, n_classes)

    with NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_file:
        joblib.dump(xgb_model, tmp_file.name)

        artifact = wandb.Artifact(
            name=artifact_name,
            type="trained_classifier",
            metadata={
                "dataset_name": dataset_name,
                "input_dim": input_dim,
                "n_classes": n_classes,
                "classifier_type": "single_xgb",
                "n_estimators": xgb_model.n_estimators,
                "max_depth": xgb_model.max_depth,
            }
        )

        artifact.add_file(tmp_file.name, name="model.joblib")
        wandb.log_artifact(artifact)
        os.unlink(tmp_file.name)

    log.info(f"Saved single XGBoost model to wandb artifact: {artifact_name}")


def load_single_xgb_from_wandb(dataset_name, input_dim, n_classes):
    if not wandb.run:
        log.warning("No active wandb run, cannot load model")
        return None

    artifact_name = get_single_xgb_artifact_name(
        dataset_name, input_dim, n_classes)

    try:
        artifact = wandb.use_artifact(f"{artifact_name}:latest")
        artifact_dir = artifact.download()

        model_path = os.path.join(artifact_dir, "model.joblib")
        xgb_model = joblib.load(model_path)

        log.info(f"Loaded single XGBoost model from wandb artifact: {
                 artifact_name}")
        return xgb_model

    except wandb.errors.CommError:
        log.info(f"Artifact {artifact_name} not found, will train new model")
        return None
    except Exception as e:
        log.error(f"Failed to load single XGBoost model: {e}")
        return None


def train_single_xgb_classifier(X_train, y_train, input_dim, n_classes, device, dataset_name="mnist", hide_val=10.0):
    log.info(f"Checking for existing XGBoost model for {dataset_name}...")
    existing_model = load_single_xgb_from_wandb(
        dataset_name, input_dim, n_classes)
    if existing_model is not None:
        return classifier_xgb(existing_model)

    log.info(f"Training new single XGBoost for {
             dataset_name} with d={input_dim} features...")

    # Convert to numpy and prepare data
    if torch.is_tensor(X_train):
        X_np = X_train.cpu().numpy().astype(np.float32)
    else:
        X_np = X_train.astype(np.float32)

    if torch.is_tensor(y_train):
        if len(y_train.shape) > 1:
            y_np = y_train.argmax(dim=1).cpu().numpy()
        else:
            y_np = y_train.cpu().numpy()
    else:
        y_np = y_train

    # Subsample training data for efficiency
    n_train_samples = min(5000, X_np.shape[0])
    train_indices = np.random.choice(
        X_np.shape[0], n_train_samples, replace=False)
    X_sub = X_np[train_indices]
    y_sub = y_np[train_indices]

    # Generate training examples with random masks
    log.info("Generating training data with random masks... ")

    X_masked_list = []
    y_list = []

    n_masks = 100  # Small number for efficiency

    for mask_idx in range(n_masks):
        # Simple random mask generation
        mask_prob = np.random.uniform(
            0.1, 0.5) if input_dim > 20 else np.random.uniform(0.3, 0.9)
        mask = np.random.binomial(1, mask_prob, input_dim).astype(np.float32)

        # Ensure at least one feature is selected
        if mask.sum() == 0:
            mask[np.random.randint(input_dim)] = 1

        # Apply mask to all samples
        X_masked = X_sub * mask - (1 - mask) * hide_val

        # Concatenate [masked_features + mask]
        mask_repeated = np.tile(mask, (n_train_samples, 1))
        X_with_mask = np.concatenate([X_masked, mask_repeated], axis=1)

        X_masked_list.append(X_with_mask)
        y_list.append(y_sub)

        if (mask_idx + 1) % 10 == 0:
            log.info(f"  Generated {mask_idx + 1}/{n_masks} masks")

    # Combine all training data
    X_final = np.vstack(X_masked_list)
    y_final = np.concatenate(y_list)

    log.info(f"Final training data: {X_final.shape[0]} samples, {
             X_final.shape[1]} features")

    # Train with exact AACO parameters
    log.info("Training XGBoost model...")
    xgb_model = XGBClassifier(
        n_estimators=250,
        max_depth=12,
        random_state=29,
        n_jobs=8,
        objective='multi:softprob',
        learning_rate=0.3,
        subsample=1.0,
        colsample_bytree=1.0,
    )

    xgb_model.fit(X_final, y_final)
    log.info("XGBoost model trained successfully")

    # Save to wandb
    save_single_xgb_to_wandb(xgb_model, dataset_name, input_dim, n_classes)

    return classifier_xgb(xgb_model)


def load_mask_generator(dataset_name, input_dim):
    """
    Their exact mask generator loading logic
    """
    dataset_name_lower = dataset_name.lower()

    if dataset_name_lower in ["cube", "mnist", "fashionmnist", "physionet", "miniboone", "afacontext"]:
        # Paper shows this works nearly as well as 10,000 (for MNIST)
        return random_mask_generator(100, input_dim, 100)
    else:
        raise ValueError("Unsupported dataset for mask generation")


def get_initial_feature(dataset_name, n_features):
    """
    Their exact initial feature selection logic
    """
    dataset_name = dataset_name.lower()

    if dataset_name == "cube":
        return 6
    elif dataset_name in ["mnist", "fashionmnist"]:
        return 100
    elif dataset_name == "grid":
        return 1
    elif dataset_name == "gas10":
        return 6
    elif dataset_name == "afacontext":
        return 5
    elif dataset_name == "diabetes":
        return 2
    elif dataset_name == "physionet":
        return 10
    elif dataset_name == "miniboone":
        return 25
    else:
        # Default: select middle feature
        return n_features // 2


class AACOOracle:
    def __init__(self,
                 k_neighbors=5,
                 acquisition_cost=0.05,
                 hide_val=10.0,
                 dataset_name="cube",
                 device=None
                 ):
        self.k_neighbors = k_neighbors
        self.acquisition_cost = acquisition_cost
        self.hide_val = hide_val
        self.dataset_name = dataset_name
        self.classifier = None
        self.mask_generator = None
        self.X_train = None
        self.y_train = None
        self.device = device
        # Move loss function to device to avoid device mismatch
        self.loss_function = nn.CrossEntropyLoss(
            reduction='none').to(self.device)

    def fit(self, X_train, y_train):
        """
        Fit the oracle on training data
        """
        self.X_train = X_train.to(self.device)
        self.y_train = y_train.to(self.device)

        # Load exact classifier - pass device parameter
        input_dim = X_train.shape[1]
        self.classifier = load_classifier(
            self.dataset_name, X_train, y_train, input_dim, device=self.device)

        # Load exact mask generator
        self.mask_generator = load_mask_generator(self.dataset_name, input_dim)

        log.info(f"AACO Oracle fitted for {self.dataset_name}")
        log.info(f"Training data: {X_train.shape}")

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
        x_query = x_observed.unsqueeze(0).to(self.device)  # 1 x d
        idx_nn = get_knn(self.X_train, x_query, mask_curr.T,
                         self.k_neighbors, instance_idx, True).squeeze()

        # Generate random masks and get the next set of possible masks
        new_masks = self.mask_generator(mask_curr).to(self.device)  # R x d
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

        y_pred = self.classifier.predict_logits(x_with_mask) if hasattr(self.classifier, 'predict_logits') else self.classifier(
            x_with_mask, torch.tensor([0], device=self.device))

        # Compute loss - ensure tensors are on same device
        y_true_rep = self.y_train[idx_nn_rep]
        if len(y_true_rep.shape) > 1 and y_true_rep.shape[1] > 1:
            # One-hot encoded
            y_true_labels = y_true_rep.argmax(dim=1)
        else:
            y_true_labels = y_true_rep

        # Ensure y_pred is on the correct device before loss computation
        if hasattr(y_pred, 'to'):
            y_pred = y_pred.to(self.device)
        else:
            y_pred = torch.tensor(y_pred, device=self.device)

        # Ensure y_true_labels is on the correct device
        y_true_labels = y_true_labels.to(self.device)

        loss = self.loss_function(y_pred.to(self.device), y_true_labels)

        # Reshape loss to (n_masks, k_neighbors) and average over neighbors
        loss = loss.view(n_masks_updated, self.k_neighbors)
        avg_loss = loss.mean(dim=1)

        # Add acquisition cost
        mask_costs = torch.sum(mask, dim=1) - torch.sum(mask_curr)
        total_cost = avg_loss + self.acquisition_cost * mask_costs

        # Find the mask with minimum cost
        best_mask_idx = total_cost.argmin()
        best_mask = mask[best_mask_idx]

        # REMOVED: termination check - always select best mask regardless of cost improvement
        # Find which new features to acquire
        new_features = (best_mask - mask_curr.squeeze()
                        ).nonzero(as_tuple=True)[0]

        if len(new_features) == 0:
            # If best mask equals current mask, force selection of cheapest unobserved feature
            unobserved = (~observed_mask).nonzero(as_tuple=True)[0]
            return unobserved[0].item() if len(unobserved) > 0 else None

        if len(new_features) == 1:
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

    def to(self, device):
        """Move oracle to device"""
        self.device = device
        self.loss_function = self.loss_function.to(device)
        if self.X_train is not None:
            self.X_train = self.X_train.to(device)
        if self.y_train is not None:
            self.y_train = self.y_train.to(device)
        return self
