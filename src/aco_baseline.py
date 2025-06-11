import torch
import logging
import warnings
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from skimage.transform import resize
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml, load_breast_cancer, load_wine

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MaskingPredictor(nn.Module):
    """
    Neural network that handles partial observations using masking strategy.
    Input: [features] + [binary_mask] indicating which features are observed.
    """

    def __init__(self, n_features, n_classes, hidden_dim=128):
        super(MaskingPredictor, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes

        # Input is features + binary mask
        input_dim = n_features * 2

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x_partial, mask):
        """
        Forward pass with partial features and mask.

        Args:
            x_partial: Partial feature vector (missing values filled with 0)
            mask: Binary mask indicating which features are observed
        """
        # Concatenate features and mask
        input_tensor = torch.cat([x_partial, mask], dim=-1)
        return self.network(input_tensor)


class ACOOracle:
    """
    Acquisition Conditioned Oracle for active feature acquisition.
    """

    def __init__(self, X_train, y_train, predictor, k_neighbors=5):
        self.X_train = X_train
        self.y_train = y_train
        self.predictor = predictor
        self.k = k_neighbors
        self.n_features = X_train.shape[1]

        # Build k-NN index for fast neighbor lookup
        self.knn_index = NearestNeighbors(n_neighbors=k_neighbors, metric="euclidean")
        self.knn_index.fit(X_train)

    def find_neighbors(self, x_observed, observed_indices):
        """
        Find k nearest neighbors based on observed features only.

        Args:
            x_observed: Values of observed features
            observed_indices: Set of indices of observed features

        Returns:
            neighbor_indices: Indices of nearest neighbors in training set
            distances: Distances to neighbors
        """
        if len(observed_indices) == 0:
            # If no features observed, return random neighbors
            n_samples = len(self.X_train)
            neighbor_indices = np.random.choice(
                n_samples, size=min(self.k, n_samples), replace=False
            )
            distances = np.ones(len(neighbor_indices))
            return neighbor_indices, distances

        # Create partial feature vector for similarity computation
        observed_list = sorted(list(observed_indices))
        x_partial = x_observed[observed_list].reshape(1, -1)
        X_train_partial = self.X_train[:, observed_list]

        # Find neighbors based on observed features only
        distances, neighbor_indices = self.knn_index.fit(X_train_partial).kneighbors(
            x_partial
        )

        return neighbor_indices[0], distances[0]

    def estimate_expected_loss(
        self, x_observed, observed_indices, candidate_feature, y_true
    ):
        """
        Estimate expected loss if we acquire the candidate feature.

        Args:
            x_observed: Current observed feature values
            observed_indices: Set of currently observed feature indices
            candidate_feature: Index of feature to potentially acquire
            y_true: True label (for loss computation)

        Returns:
            Expected cross-entropy loss
        """
        # Find neighbors based on currently observed features
        neighbor_indices, _ = self.find_neighbors(x_observed, observed_indices)

        total_loss = 0.0

        for neighbor_idx in neighbor_indices:
            # Get the value of candidate feature from this neighbor
            candidate_value = self.X_train[neighbor_idx, candidate_feature]

            # Create new observed feature vector including candidate
            new_observed_indices = observed_indices | {candidate_feature}
            x_new = np.zeros(self.n_features)

            # Fill in observed values
            for idx in observed_indices:
                x_new[idx] = x_observed[idx]
            x_new[candidate_feature] = candidate_value

            # Create mask
            mask = np.zeros(self.n_features)
            for idx in new_observed_indices:
                mask[idx] = 1.0

            # Compute prediction and loss
            with torch.no_grad():
                x_tensor = (
                    torch.tensor(x_new, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                )
                mask_tensor = (
                    torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                )
                y_tensor = torch.tensor([y_true], dtype=torch.long).to(DEVICE)

                logits = self.predictor(x_tensor, mask_tensor)
                loss = nn.CrossEntropyLoss()(logits, y_tensor)
                total_loss += loss.item()

        return total_loss / len(neighbor_indices)

    def greedy_select_feature(
        self, x_observed, observed_indices, remaining_candidates, y_true
    ):
        """
        Greedily select the next best feature to acquire.

        Args:
            x_observed: Current observed feature values
            observed_indices: Set of currently observed feature indices
            remaining_candidates: Set of features not yet acquired
            y_true: True label

        Returns:
            Index of selected feature
        """
        best_feature = None
        best_loss = float("inf")

        for candidate in remaining_candidates:
            expected_loss = self.estimate_expected_loss(
                x_observed, observed_indices, candidate, y_true
            )

            if expected_loss < best_loss:
                best_loss = expected_loss
                best_feature = candidate

        return best_feature

    def full_aco_select_feature(
        self, x_observed, observed_indices, remaining_candidates, y_true, n_subsets=1000
    ):
        """
        Full ACO: consider joint acquisition of multiple features, then select one.

        Args:
            x_observed: Current observed feature values
            observed_indices: Set of currently observed feature indices
            remaining_candidates: Set of features not yet acquired
            y_true: True label
            n_subsets: Number of random subsets to consider

        Returns:
            Index of selected feature
        """
        if len(remaining_candidates) == 0:
            return None

        best_subset = None
        best_loss = float("inf")

        # Consider subsets of remaining candidates
        candidates_list = list(remaining_candidates)

        for _ in range(n_subsets):
            # Sample random subset size (1 to min(5, len(candidates)))
            max_subset_size = min(5, len(candidates_list))
            subset_size = np.random.randint(1, max_subset_size + 1)

            # Sample random subset
            subset = set(
                np.random.choice(candidates_list, size=subset_size, replace=False)
            )

            # Estimate expected loss for this subset
            expected_loss = self.estimate_subset_loss(
                x_observed, observed_indices, subset, y_true
            )

            if expected_loss < best_loss:
                best_loss = expected_loss
                best_subset = subset

        # From best subset, select one feature uniformly at random
        if best_subset:
            return np.random.choice(list(best_subset))
        else:
            return np.random.choice(candidates_list)

    def estimate_subset_loss(
        self, x_observed, observed_indices, candidate_subset, y_true
    ):
        """
        Estimate expected loss if we acquire all features in candidate_subset.
        """
        neighbor_indices, _ = self.find_neighbors(x_observed, observed_indices)

        total_loss = 0.0

        for neighbor_idx in neighbor_indices:
            # Create new observed feature vector including all candidates
            new_observed_indices = observed_indices | candidate_subset
            x_new = np.zeros(self.n_features)

            # Fill in observed values
            for idx in observed_indices:
                x_new[idx] = x_observed[idx]

            # Fill in candidate values from neighbor
            for idx in candidate_subset:
                x_new[idx] = self.X_train[neighbor_idx, idx]

            # Create mask
            mask = np.zeros(self.n_features)
            for idx in new_observed_indices:
                mask[idx] = 1.0

            # Compute prediction and loss
            with torch.no_grad():
                x_tensor = (
                    torch.tensor(x_new, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                )
                mask_tensor = (
                    torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                )
                y_tensor = torch.tensor([y_true], dtype=torch.long).to(DEVICE)

                logits = self.predictor(x_tensor, mask_tensor)
                loss = nn.CrossEntropyLoss()(logits, y_tensor)
                total_loss += loss.item()

        return total_loss / len(neighbor_indices)


def load_dataset(dataset_name):
    """
    Load and preprocess dataset.

    Args:
        dataset_name: Name of dataset ('mnist', 'breast_cancer', 'wine')

    Returns:
        X: Feature matrix
        y: Labels
        n_classes: Number of classes
    """
    logger.info(f"Loading {dataset_name} dataset...")

    if dataset_name == "mnist":
        # Load MNIST and downsample to 16x16 like the paper
        mnist = fetch_openml("mnist_784", version=1)
        X, y = mnist.data, mnist.target.astype(int)

        # Convert to numpy and reshape
        X = X.values if hasattr(X, "values") else X
        X = X.reshape(-1, 28, 28)

        # Downsample to 16x16
        X_downsampled = []
        for img in X:
            img_resized = resize(img, (16, 16), anti_aliasing=True)
            X_downsampled.append(img_resized.flatten())

        X = np.array(X_downsampled)
        y = np.array(y)
        n_classes = 10

        # Use subset for faster experimentation
        n_samples = 10000
        indices = np.random.choice(len(X), n_samples, replace=False)
        X, y = X[indices], y[indices]

    elif dataset_name == "breast_cancer":
        data = load_breast_cancer()
        X, y = data.data, data.target
        n_classes = 2

    elif dataset_name == "wine":
        data = load_wine()
        X, y = data.data, data.target
        n_classes = 3

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    logger.info(
        f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features, {
            n_classes
        } classes"
    )
    return X, y, n_classes


def train_masking_predictor(X_train, y_train, n_classes, epochs=50):
    """
    Train the masking predictor on training data with random feature masks.
    """
    logger.info("Training masking predictor...")

    n_features = X_train.shape[1]
    model = MaskingPredictor(n_features, n_classes).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Create training data with random masks
    n_samples = len(X_train)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # Create random batches with random masks
        batch_size = 64
        n_batches = (n_samples + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)

            batch_X = X_train[start_idx:end_idx]
            batch_y = y_train[start_idx:end_idx]

            # Create random masks (observe 20-80% of features)
            batch_masks = []
            batch_X_masked = []

            for x in batch_X:
                n_observed = np.random.randint(
                    int(0.2 * n_features), int(0.8 * n_features) + 1
                )
                observed_indices = np.random.choice(
                    n_features, n_observed, replace=False
                )

                mask = np.zeros(n_features)
                mask[observed_indices] = 1.0

                x_masked = x.copy()
                x_masked[mask == 0] = 0  # Zero out unobserved features

                batch_masks.append(mask)
                batch_X_masked.append(x_masked)

            # Convert to tensors
            X_tensor = torch.tensor(np.array(batch_X_masked), dtype=torch.float32).to(
                DEVICE
            )
            mask_tensor = torch.tensor(np.array(batch_masks), dtype=torch.float32).to(
                DEVICE
            )
            y_tensor = torch.tensor(batch_y, dtype=torch.long).to(DEVICE)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_tensor, mask_tensor)
            loss = criterion(outputs, y_tensor)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / n_batches
            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    logger.info("Masking predictor training completed")
    return model


def run_aco_experiment(X_test, y_test, oracle, max_features=10, method="greedy"):
    """
    Run ACO experiment on test set.

    Args:
        X_test: Test features
        y_test: Test labels
        oracle: Trained ACO oracle
        max_features: Maximum number of features to acquire per instance
        method: 'greedy' or 'full' ACO

    Returns:
        Dictionary with results
    """
    logger.info(f"Running ACO experiment with {method} method...")

    results = {"accuracies": [], "n_features_acquired": [], "final_accuracy": None}

    n_features = X_test.shape[1]
    n_correct = 0

    for i, (x_true, y_true) in enumerate(tqdm(zip(X_test, y_test), total=len(X_test))):
        observed_indices = set()
        x_observed = np.zeros(n_features)

        # Acquire features sequentially
        for step in range(max_features):
            remaining_candidates = set(range(n_features)) - observed_indices

            if len(remaining_candidates) == 0:
                break

            # Select next feature
            if method == "greedy":
                next_feature = oracle.greedy_select_feature(
                    x_observed, observed_indices, remaining_candidates, y_true
                )
            elif method == "full":
                next_feature = oracle.full_aco_select_feature(
                    x_observed, observed_indices, remaining_candidates, y_true
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            if next_feature is None:
                break

            # Acquire the selected feature
            observed_indices.add(next_feature)
            x_observed[next_feature] = x_true[next_feature]

        # Make final prediction
        mask = np.zeros(n_features)
        for idx in observed_indices:
            mask[idx] = 1.0

        with torch.no_grad():
            x_tensor = (
                torch.tensor(x_observed, dtype=torch.float32)
                .unsqueeze(0)
                .to(oracle.device)
            )
            mask_tensor = (
                torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(oracle.device)
            )

            logits = oracle.predictor(x_tensor, mask_tensor)
            prediction = torch.argmax(logits, dim=1).item()

            if prediction == y_true:
                n_correct += 1

        # Record results for this instance
        current_accuracy = n_correct / (i + 1)
        results["accuracies"].append(current_accuracy)
        results["n_features_acquired"].append(len(observed_indices))

        if i % 100 == 0:
            logger.info(
                f"Processed {i} instances, current accuracy: {current_accuracy:.3f}"
            )

    results["final_accuracy"] = n_correct / len(X_test)

    logger.info(f"Final accuracy: {results['final_accuracy']:.3f}")
    logger.info(
        f"Average features acquired: {np.mean(results['n_features_acquired']):.1f}"
    )

    return results


def plot_results(results_greedy, results_full, dataset_name):
    """
    Plot comparison of greedy vs full ACO methods.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy over time
    ax1.plot(results_greedy["accuracies"], label="Greedy ACO", alpha=0.8)
    ax1.plot(results_full["accuracies"], label="Full ACO", alpha=0.8)
    ax1.set_xlabel("Test Instance")
    ax1.set_ylabel("Cumulative Accuracy")
    ax1.set_title(f"Accuracy Progression - {dataset_name}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Features acquired histogram
    ax2.hist(
        results_greedy["n_features_acquired"], alpha=0.7, label="Greedy ACO", bins=20
    )
    ax2.hist(results_full["n_features_acquired"], alpha=0.7, label="Full ACO", bins=20)
    ax2.set_xlabel("Number of Features Acquired")
    ax2.set_ylabel("Frequency")
    ax2.set_title(f"Features Acquired Distribution - {dataset_name}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"aco_results_{dataset_name}.png", dpi=300, bbox_inches="tight")
    plt.show()


def main(args):
    logger.info(f"Using device: {DEVICE}")

    # Load dataset
    X, y, n_classes = load_dataset(args.dataset)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y
    )

    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Train masking predictor
    predictor = train_masking_predictor(X_train, y_train, n_classes)

    # Create ACO oracle
    oracle = ACOOracle(X_train, y_train, predictor, args.k_neighbors)

    # Run experiments
    results_greedy = run_aco_experiment(
        X_test[:100], y_test[:100], oracle, args.max_features, method="greedy"
    )

    results_full = run_aco_experiment(
        X_test[:100], y_test[:100], oracle, args.max_features, method="full"
    )

    # Plot results
    plot_results(results_greedy, results_full, args.dataset)

    # Print summary
    print(f"\n=== Results Summary for {args.dataset} ===")
    print(f"Greedy ACO - Final Accuracy: {results_greedy['final_accuracy']:.3f}")
    print(
        f"Greedy ACO - Avg Features: {
            np.mean(results_greedy['n_features_acquired']):.1f
        }"
    )
    print(f"Full ACO - Final Accuracy: {results_full['final_accuracy']:.3f}")
    print(
        f"Full ACO - Avg Features: {np.mean(results_full['n_features_acquired']):.1f}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ACO Baseline Implementation")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "breast_cancer", "wine"],
        default="mnist",
        help="Dataset to use",
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=10,
        help="Maximum number of features to acquire per instance",
    )
    parser.add_argument(
        "--k_neighbors", type=int, default=5, help="Number of neighbors for k-NN"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.3,
        help="Fraction of data to use for testing",
    )

    args = parser.parse_args()

    main(args)
