import torch
import logging

log = logging.getLogger(__name__)


class classifier_mlp:
    """MLP classifier wrapper that adapts WrappedMaskedMLPClassifier to AACO interface
    Handles the conversion between AACO's hide_val masking and MLP's zero masking
    """

    def __init__(self, wrapped_mlp_classifier, hide_val=10.0):
        self.wrapped_mlp = wrapped_mlp_classifier
        self.hide_val = hide_val

    def predict_logits(self, X):
        """AACO interface expects this method for compatibility

        Args:
            X: tensor of shape [batch_size, features + mask] where features use hide_val for missing
        Returns:
            logits: tensor of shape [batch_size, n_classes]

        """
        # Split features and mask
        n_features = X.shape[1] // 2
        features_with_hide_val = X[:, :n_features]
        mask = X[:, n_features:]

        # Convert from AACO's hide_val masking to MLP's zero masking
        # Where mask=0, AACO uses hide_val, but MLP expects 0
        masked_features = (
            features_with_hide_val * mask
        )  # This zeros out unobserved features

        # Use the MLP classifier
        with torch.no_grad():
            logits = self.wrapped_mlp.module(masked_features, mask)

        return logits

    def __call__(self, X, idx):
        """AACO interface for compatibility
        """
        # Get logits and convert to probabilities for AACO compatibility
        logits = self.predict_logits(X)
        probs = torch.softmax(logits, dim=-1)

        return probs
