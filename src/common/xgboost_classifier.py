import torch
import pickle
import numpy as np
import xgboost as xgb
import lightning as pl

from pathlib import Path
from typing import Self, final, override
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from common.custom_types import (
    AFAClassifier,
    FeatureMask,
    Label,
    MaskedFeatures,
)


@final
class XGBoostAFAClassifier(AFAClassifier):
    """XGBoost classifier that implements the AFAClassifier interface."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        model: xgb.XGBClassifier | None = None,
        device: torch.device = torch.device("cpu"),
        **xgb_params
    ):
        self.n_features = n_features
        self.n_classes = n_classes
        self._device = device
        self.label_encoder = LabelEncoder()

        # Default XGBoost parameters matching the ACO paper
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist',
            'enable_categorical': False,
        }

        # Override defaults with provided parameters
        default_params.update(xgb_params)

        if model is None:
            if n_classes == 2:
                self.model = xgb.XGBClassifier(
                    objective='binary:logistic',
                    **default_params
                )
            else:
                self.model = xgb.XGBClassifier(
                    objective='multi:softprob',
                    **default_params
                )
        else:
            self.model = model

        self.is_fitted = False

    def fit(self, X: torch.Tensor, y: torch.Tensor, feature_mask: torch.Tensor):
        X_np = X.cpu().numpy()
        y_np = y.cpu().numpy()
        feature_mask_np = feature_mask.cpu().numpy()

        # Prepare training data with feature mask information
        # Concatenate features and feature mask like MaskedMLP
        X_train = np.concatenate(
            [X_np, feature_mask_np.astype(np.float32)], axis=1)

        # Multi-class: convert from one-hot to class indices
        if y_np.ndim > 1 and y_np.shape[1] > 1:
            y_train = np.argmax(y_np, axis=1)

        # Binary or already class indices
        else:
            y_train = y_np.squeeze()

        self.label_encoder.fit(y_train)
        y_train = self.label_encoder.transform(y_train)

        self.model.fit(X_train, y_train)
        self.is_fitted = True

    @override
    def __call__(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
    ) -> Label:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        masked_features_np = masked_features.cpu().numpy()
        feature_mask_np = feature_mask.cpu().numpy().astype(np.float32)

        # Prepare input data (concatenate features and mask)
        X_input = np.concatenate([masked_features_np, feature_mask_np], axis=1)

        # Binary classification
        if self.n_classes == 2:
            probs = self.model.predict_proba(X_input)
            if probs.shape[1] == 2:
                probs = probs[:, 1:2]  # Take positive class probability
            probs = np.column_stack([1 - probs.squeeze(), probs.squeeze()])

        # Multi-class classification
        else:
            probs = self.model.predict_proba(X_input)

        # Convert back to torch tensor
        probs_tensor = torch.from_numpy(probs).float().to(self._device)

        return probs_tensor

    @override
    def save(self, path: Path) -> None:
        """Save the XGBoost classifier."""
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'model': self.model,
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'label_encoder': self.label_encoder,
            'is_fitted': self.is_fitted,
        }

        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)

    @override
    @classmethod
    def load(cls, path: Path, device: torch.device) -> Self:
        """Load the XGBoost classifier."""
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)

        classifier = cls(
            n_features=save_dict['n_features'],
            n_classes=save_dict['n_classes'],
            model=save_dict['model'],
            device=device,
        )
        classifier.label_encoder = save_dict['label_encoder']
        classifier.is_fitted = save_dict['is_fitted']

        return classifier

    @override
    def to(self, device: torch.device) -> Self:
        """Move to device (XGBoost runs on CPU, but we track device for tensors)."""
        self._device = device
        return self

    @property
    @override
    def device(self) -> torch.device:
        """Get current device."""
        return self._device


@final
class LitXGBoostAFAClassifier(pl.LightningModule):
    """Lightning module for training XGBoost classifier."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        min_masking_probability: float = 0.0,
        max_masking_probability: float = 1.0,
        **xgb_params
    ):
        super().__init__()

        self.n_features = n_features
        self.n_classes = n_classes
        self.min_masking_probability = min_masking_probability
        self.max_masking_probability = max_masking_probability

        self.classifier = XGBoostAFAClassifier(
            n_features=n_features,
            n_classes=n_classes,
            **xgb_params
        )

        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        """Setup method called by Lightning."""
        if stage == "fit":
            # Collect all training data to fit XGBoost
            train_dataloader = self.trainer.datamodule.train_dataloader()

            all_features = []
            all_labels = []
            all_masks = []

            for batch in train_dataloader:
                features, labels = batch

                # Apply random masking for training
                batch_size = features.shape[0]
                masking_prob = torch.rand(
                    batch_size, 1, device=features.device)
                masking_prob = (masking_prob * (self.max_masking_probability - self.min_masking_probability) +
                                self.min_masking_probability)

                # Create random masks
                mask_probs = torch.rand_like(features)
                feature_mask = mask_probs > masking_prob

                # Apply masks
                masked_features = features * feature_mask.float()

                all_features.append(masked_features)
                all_labels.append(labels)
                all_masks.append(feature_mask)

            # Concatenate all data
            X_train = torch.cat(all_features, dim=0)
            y_train = torch.cat(all_labels, dim=0)
            masks_train = torch.cat(all_masks, dim=0)

            # Fit the XGBoost model
            self.classifier.fit(X_train, y_train, masks_train)

    def training_step(self, batch, batch_idx):
        """Training step (XGBoost is already fitted in setup)."""
        return torch.tensor(0.0, requires_grad=True)  # Dummy loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        features, labels = batch

        # Test with many observations
        masking_prob = torch.full(
            (features.shape[0], 1), self.min_masking_probability, device=features.device
        )
        mask_probs = torch.rand_like(features)
        feature_mask_many = mask_probs > masking_prob
        masked_features_many = features * feature_mask_many.float()

        if self.classifier.is_fitted:
            pred_probs_many = self.classifier(
                masked_features_many, feature_mask_many)
            loss_many = torch.nn.functional.cross_entropy(
                pred_probs_many, labels.argmax(dim=1))
            acc_many = accuracy_score(
                labels.argmax(dim=1).cpu().numpy(),
                pred_probs_many.argmax(dim=1).cpu().numpy()
            )
        else:
            loss_many = torch.tensor(float('inf'))
            acc_many = 0.0

        # Test with few observations
        masking_prob = torch.full(
            (features.shape[0], 1), self.max_masking_probability, device=features.device
        )
        mask_probs = torch.rand_like(features)
        feature_mask_few = mask_probs > masking_prob
        masked_features_few = features * feature_mask_few.float()

        if self.classifier.is_fitted:
            pred_probs_few = self.classifier(
                masked_features_few, feature_mask_few)
            loss_few = torch.nn.functional.cross_entropy(
                pred_probs_few, labels.argmax(dim=1))
            acc_few = accuracy_score(
                labels.argmax(dim=1).cpu().numpy(),
                pred_probs_few.argmax(dim=1).cpu().numpy()
            )
        else:
            loss_few = torch.tensor(float('inf'))
            acc_few = 0.0

        self.log("val_loss_many_observations", loss_many)
        self.log("val_acc_many_observations", acc_many)
        self.log("val_loss_few_observations", loss_few)
        self.log("val_acc_few_observations", acc_few)

        return loss_many

    def configure_optimizers(self):
        """Dummy optimizer since XGBoost doesn't use gradient descent."""
        return torch.optim.Adam([torch.tensor(0.0, requires_grad=True)], lr=1e-3)


@final
class WrappedXGBoostAFAClassifier(AFAClassifier):
    """Wrapped XGBoost classifier for the AFA pipeline."""

    def __init__(self, lit_model: LitXGBoostAFAClassifier, device: torch.device):
        self.lit_model = lit_model
        self.classifier = lit_model.classifier
        self._device = device

    @override
    def __call__(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
    ) -> Label:
        """Predict class probabilities."""
        return self.classifier(masked_features, feature_mask)

    @override
    def save(self, path: Path) -> None:
        """Save the classifier."""
        self.classifier.save(path)

    @override
    @classmethod
    def load(cls, path: Path, device: torch.device) -> Self:
        """Load the classifier."""
        classifier = XGBoostAFAClassifier.load(path, device)

        # Create a dummy lit_model for interface compatibility
        lit_model = LitXGBoostAFAClassifier(
            n_features=classifier.n_features,
            n_classes=classifier.n_classes,
        )
        lit_model.classifier = classifier

        return cls(lit_model=lit_model, device=device)

    @override
    def to(self, device: torch.device) -> Self:
        """Move to device."""
        self._device = device
        self.classifier.to(device)
        return self

    @property
    @override
    def device(self) -> torch.device:
        """Get current device."""
        return self._device
