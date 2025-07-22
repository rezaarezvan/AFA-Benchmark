import torch
import pickle
import numpy as np
import xgboost as xgb
import lightning as pl

from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from typing import Self, final, override, Optional

from common.custom_types import (
    AFAClassifier,
    FeatureMask,
    Label,
    MaskedFeatures,
)


@final
class XGBoostAFAClassifier(AFAClassifier):
    """XGBoost classifier that implements the AFAClassifier interface with dictionary approach for low-dim."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        model: xgb.XGBClassifier | None = None,
        device: torch.device = torch.device("cpu"),
        use_dictionary_approach: bool = True,
        dictionary_threshold: int = 12,
        subsample_ratio: float = 0.8,
        **xgb_params
    ):
        self.n_features = n_features
        self.n_classes = n_classes
        self._device = device
        self.label_encoder = LabelEncoder()
        self.use_dictionary_approach = use_dictionary_approach and n_features <= dictionary_threshold
        self.subsample_ratio = subsample_ratio

        # Dictionary for storing models per mask pattern (original AACO approach)
        self.xgb_model_dict = {}
        self.X_train_numpy = None
        self.y_train_numpy = None

        # Original AACO parameters
        default_params = {
            'n_estimators': 250,  # Original AACO value
            'max_depth': 5,       # Original AACO value
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist',
            'enable_categorical': False,
        }

        # Override defaults with provided parameters
        default_params.update(xgb_params)

        # Single model for high-dimensional masking approach
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

        # Store training data for dictionary approach
        if self.use_dictionary_approach:
            self.X_train_numpy = X_np
            # Convert from one-hot to class indices
            if y_np.ndim > 1 and y_np.shape[1] > 1:
                self.y_train_numpy = np.argmax(y_np, axis=1)
            else:
                self.y_train_numpy = y_np.squeeze()
            self.is_fitted = True
            return

        # Original masking approach for high dimensions
        # Prepare training data with feature mask information
        # Concatenate features and feature mask like MaskedMLP
        X_train = np.concatenate(
            [X_np, feature_mask_np.astype(np.float32)], axis=1)

        # Multi-class: convert from one-hot to class indices
        if y_np.ndim > 1 and y_np.shape[1] > 1:
            y_train = np.argmax(y_np, axis=1)
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
        neighbor_indices: Optional[torch.Tensor] = None
    ) -> Label:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Convert to numpy
        masked_features_np = masked_features.cpu().numpy()
        feature_mask_np = feature_mask.cpu().numpy()

        if self.use_dictionary_approach:
            # Original AACO dictionary approach
            # Input format: [features, mask] concatenated
            # Extract features and mask
            if masked_features.shape[1] == self.n_features * 2:
                # Already concatenated format
                X_concat = masked_features_np
            else:
                # Need to concatenate
                X_concat = np.concatenate(
                    [masked_features_np, feature_mask_np], axis=1)

            return self._predict_dictionary(X_concat)
        else:
            # High-dimensional masking approach
            X_concat = np.concatenate(
                [masked_features_np, feature_mask_np], axis=1)
            predictions = self.model.predict_proba(X_concat)
            return torch.tensor(predictions, dtype=torch.float32, device=self._device)

    def _predict_dictionary(self, X_concat: np.ndarray) -> torch.Tensor:
        """Dictionary-based prediction matching original AACO implementation."""
        n = X_concat.shape[0]
        probs = torch.zeros((n, self.n_classes), device=self._device)

        for i in range(n):
            # Extract mask and convert to string key (original AACO approach)
            mask_i = X_concat[i][self.n_features:]
            nonzero_i = np.nonzero(mask_i)[0]
            mask_i_string = ''.join(map(str, mask_i.astype(int).tolist()))

            # Train new model if mask pattern not seen before
            if mask_i_string not in self.xgb_model_dict:
                # Original AACO parameters
                self.xgb_model_dict[mask_i_string] = xgb.XGBClassifier(
                    n_estimators=250,
                    max_depth=5,
                    random_state=42,
                    n_jobs=-1
                )

                # Train on subset of features for this mask
                if len(nonzero_i) > 0:
                    X_train_subset = self.X_train_numpy[:, nonzero_i]
                    # Subsample training data (original AACO approach)
                    n_samples = int(
                        self.X_train_numpy.shape[0] * self.subsample_ratio)
                    idx = np.random.choice(
                        self.X_train_numpy.shape[0], n_samples, replace=False)
                    self.xgb_model_dict[mask_i_string].fit(
                        X_train_subset[idx], self.y_train_numpy[idx]
                    )
                else:
                    # No features selected - create dummy prediction
                    dummy_probs = np.ones(self.n_classes) / self.n_classes
                    probs[i] = torch.tensor(
                        dummy_probs, dtype=torch.float32, device=self._device)
                    continue

            # Predict using only the relevant features (no imputation!)
            if len(nonzero_i) > 0:
                X_relevant = X_concat[i, nonzero_i].reshape(1, -1)
                pred_probs = self.xgb_model_dict[mask_i_string].predict_proba(
                    X_relevant)
                probs[i] = torch.tensor(
                    pred_probs[0], dtype=torch.float32, device=self._device)
            else:
                # No features - uniform prediction
                dummy_probs = np.ones(self.n_classes) / self.n_classes
                probs[i] = torch.tensor(
                    dummy_probs, dtype=torch.float32, device=self._device)

        return probs

    @override
    def save(self, path: Path) -> None:
        """Save the classifier."""
        path.mkdir(exist_ok=True)

        if self.use_dictionary_approach:
            # Save dictionary of models
            with open(path / "xgb_model_dict.pkl", "wb") as f:
                pickle.dump(self.xgb_model_dict, f)

            # Save training data
            if self.X_train_numpy is not None:
                np.save(path / "X_train.npy", self.X_train_numpy)
                np.save(path / "y_train.npy", self.y_train_numpy)
        else:
            # Save single model
            with open(path / "xgb_model.pkl", "wb") as f:
                pickle.dump(self.model, f)

            # Save label encoder
            with open(path / "label_encoder.pkl", "wb") as f:
                pickle.dump(self.label_encoder, f)

        # Save metadata
        metadata = {
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'use_dictionary_approach': self.use_dictionary_approach,
            'subsample_ratio': self.subsample_ratio,
            'is_fitted': self.is_fitted
        }
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        """Load the classifier."""
        # Load metadata
        with open(path / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        # Create instance
        instance = cls(
            n_features=metadata['n_features'],
            n_classes=metadata['n_classes'],
            device=device,
            use_dictionary_approach=metadata['use_dictionary_approach'],
            subsample_ratio=metadata['subsample_ratio']
        )

        if metadata['use_dictionary_approach']:
            # Load dictionary of models
            with open(path / "xgb_model_dict.pkl", "rb") as f:
                instance.xgb_model_dict = pickle.load(f)

            # Load training data if exists
            if (path / "X_train.npy").exists():
                instance.X_train_numpy = np.load(path / "X_train.npy")
                instance.y_train_numpy = np.load(path / "y_train.npy")
        else:
            # Load single model
            with open(path / "xgb_model.pkl", "rb") as f:
                instance.model = pickle.load(f)

            # Load label encoder
            with open(path / "label_encoder.pkl", "rb") as f:
                instance.label_encoder = pickle.load(f)

        instance.is_fitted = metadata['is_fitted']
        return instance

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
    """Lightning wrapper for XGBoost classifier to maintain compatibility."""

    def __init__(
        self,
        classifier: XGBoostAFAClassifier,
        min_masking_probability: float = 0.0,
        max_masking_probability: float = 0.9,
    ):
        super().__init__()
        self.classifier = classifier
        self.min_masking_probability = min_masking_probability
        self.max_masking_probability = max_masking_probability

    def training_step(self, batch, batch_idx):
        """Training step - XGBoost trains during fit(), not here."""
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
        neighbor_indices: Optional[torch.Tensor] = None
    ) -> Label:
        """Predict class probabilities."""
        return self.classifier(masked_features, feature_mask, neighbor_indices)

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
        lit_model = LitXGBoostAFAClassifier(classifier=classifier)
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
