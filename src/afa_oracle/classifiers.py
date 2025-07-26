import torch
import logging
import numpy as np
import torch.nn as nn

from scipy.stats import norm

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

class classifier_mlp():
    """
    MLP classifier wrapper that adapts WrappedMaskedMLPClassifier to AACO interface
    Handles the conversion between AACO's hide_val masking and MLP's zero masking
    """

    def __init__(self, wrapped_mlp_classifier, hide_val=10.0):
        self.wrapped_mlp = wrapped_mlp_classifier
        self.hide_val = hide_val

    def predict_logits(self, X):
        """
        AACO interface expects this method for compatibility

        Args:
            X: tensor of shape [batch_size, features + mask] where features use hide_val for missing
        Returns:
            logits: tensor of shape [batch_size, n_classes]
        """
        device = X.device if hasattr(X, 'device') else torch.device('cpu')

        # Split features and mask
        n_features = X.shape[1] // 2
        features_with_hide_val = X[:, :n_features]
        mask = X[:, n_features:]

        # Convert from AACO's hide_val masking to MLP's zero masking
        # Where mask=0, AACO uses hide_val, but MLP expects 0
        masked_features = features_with_hide_val * \
            mask  # This zeros out unobserved features

        # Use the MLP classifier
        with torch.no_grad():
            logits = self.wrapped_mlp.module(masked_features, mask)

        return logits

    def __call__(self, X, idx):
        """
        AACO interface for compatibility
        """
        device = X.device if hasattr(X, 'device') else torch.device('cpu')

        # Get logits and convert to probabilities for AACO compatibility
        logits = self.predict_logits(X)
        probs = torch.softmax(logits, dim=-1)

        return probs
