import torch
import torch.nn
import numpy as np

from scipy.stats import norm
from distutils.log import error
from xgboost import XGBClassifier


class NaiveBayes(torch.nn.Module):
    """
    Their exact NaiveBayes implementation for CUBE dataset
    """

    def __init__(self, num_features, num_classes, std):
        super(NaiveBayes, self).__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.std = std

    def forward(self, x):
        try:
            mask = x[:, self.num_features:]
            x = x[:, :self.num_features]
        except IndexError:
            error(
                "Classifier expects masking information to be concatenated with each feature vector.")

        y_classes = list(range(self.num_classes))
        output_probs = torch.zeros((len(x), self.num_classes))

        for y_val in y_classes:
            # PDF values for each feature in x conditioned on the given label y_val

            # Default to PDF for U[0,1)
            p_x_y = torch.where((x >= 0) & (x < 1), torch.ones(
                x.shape), torch.zeros(x.shape))

            # Use normal distribution PDFs for appropriate features given y_val
            p_x_y[:, y_val:y_val+3] = torch.transpose(
                torch.Tensor(np.array([norm.pdf(x[:, y_val], y_val % 2, self.std),
                                       norm.pdf(x[:, y_val+1],
                                                (y_val // 2) % 2, self.std),
                                       norm.pdf(x[:, y_val+2], (y_val // 4) % 2, self.std)])), 0, 1)

            # Compute joint probability over masked features
            p_xo_y = torch.prod(torch.where(
                torch.gt(mask, 0), p_x_y, torch.tensor(1).float()), dim=1)

            p_y = 1 / self.num_classes
            output_probs[:, y_val] = p_xo_y * p_y

        # Add numerical stability: prevent division by zero
        prob_sums = torch.sum(output_probs, axis=1)
        # Prevent division by zero
        prob_sums = torch.clamp(prob_sums, min=1e-10)

        # Check for any remaining numerical issues
        if torch.any(torch.isnan(prob_sums)) or torch.any(torch.isinf(prob_sums)):
            # Return uniform distribution as fallback
            uniform_probs = torch.ones_like(output_probs) / self.num_classes
            return uniform_probs

        normalized_probs = output_probs / prob_sums.unsqueeze(1)

        # Final safety check for NaN/inf
        if torch.any(torch.isnan(normalized_probs)) or torch.any(torch.isinf(normalized_probs)):
            # Return uniform distribution as fallback
            uniform_probs = torch.ones_like(output_probs) / self.num_classes
            return uniform_probs

        return normalized_probs

    def predict(self, x):
        return self.forward(x)


class classifier_xgb_dict():
    """
    Their XGBoost dictionary classifier for dynamic training
    """

    def __init__(self, output_dim, input_dim, subsample_ratio, X_train, y_train):
        """
        Input:
        output_dim: Dimension of the outcome y
        input_dim: Dimension of the input features (X)
        subsample_ratio: Fraction of training points for each boosting iteration
        """
        self.xgb_model_dict = {}
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.subsample_ratio = subsample_ratio
        self.X_train_numpy = X_train.numpy()
        self.y_train_numpy = y_train.argmax(dim=1).numpy()

    def __call__(self, X, idx):
        n = X.shape[0]
        probs = torch.zeros((n, self.output_dim))

        for i in range(n):
            # Which mask?
            mask_i = X[i][self.input_dim:]
            nonzero_i = mask_i.nonzero().squeeze()

            # Handle edge case: ensure nonzero_i is always an array
            if nonzero_i.dim() == 0:  # scalar case (single feature)
                nonzero_i = nonzero_i.unsqueeze(0)

            # Check if no features are selected
            if len(nonzero_i) == 0:
                # No features selected - return uniform probabilities
                dummy_probs = torch.ones(self.output_dim) / self.output_dim
                probs[i] = dummy_probs
                continue

            mask_i_string = ''.join(map(str, mask_i.long().tolist()))

            # Is the mask in the dictionary?
            if mask_i_string not in self.xgb_model_dict:
                self.xgb_model_dict[mask_i_string] = XGBClassifier(
                    n_estimators=250, max_depth=5, random_state=29, n_jobs=8)

                # Extract features for selected indices
                X_train_subset = self.X_train_numpy[:, nonzero_i.cpu().numpy()]

                # Ensure we have the right shape
                if X_train_subset.ndim == 1:
                    X_train_subset = X_train_subset.reshape(-1, 1)

                # Subsample training data
                n_samples = max(
                    1, int(X_train_subset.shape[0] * self.subsample_ratio))
                idx_sample = np.random.choice(
                    X_train_subset.shape[0], n_samples, replace=False)

                self.xgb_model_dict[mask_i_string].fit(
                    X_train_subset[idx_sample], self.y_train_numpy[idx_sample])

            # Prediction
            X_query = X[i, nonzero_i].cpu().numpy().reshape(1, -1)
            pred_probs = self.xgb_model_dict[mask_i_string].predict_proba(
                X_query)
            probs[i] = torch.from_numpy(pred_probs[0])

        return probs


class classifier_ground_truth():
    """
    Wrapper for their ground truth NaiveBayes classifier
    """

    def __init__(self, num_features=20, num_classes=8, std=0.3):
        self.gt_classifier = NaiveBayes(
            num_features=num_features, num_classes=num_classes, std=std)

    def __call__(self, X, idx):
        return self.gt_classifier.predict(X)


class classifier_xgb():
    """
    Wrapper for pre-trained XGBoost models
    """

    def __init__(self, xgb_model):
        self.xgb_model = xgb_model

    def __call__(self, X, idx):
        return torch.tensor(self.xgb_model.predict_proba(X))
