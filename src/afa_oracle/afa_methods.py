import torch
import logging
import torch.nn as nn

from pathlib import Path
from dataclasses import dataclass
from typing import Self, final, override, List, Tuple
from torch.utils.data import DataLoader, TensorDataset

from .aco_core import ACOOracle, create_aco_oracle
from common.custom_types import (
    AFAMethod, AFAClassifier, AFASelection, FeatureMask,
    Label, MaskedFeatures, AFADataset
)

logger = logging.getLogger(__name__)


@dataclass
@final
class ACOAFAMethod(AFAMethod):
    """
    Non-parametric ACO method implementation.
    """

    aco_oracle: ACOOracle
    afa_classifier: AFAClassifier
    _device: torch.device = torch.device("cpu")

    def __post_init__(self):
        self.afa_classifier = self.afa_classifier.to(self._device)

    @override
    def select(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> AFASelection:
        """
        Select next feature to acquire using ACO oracle.
        """

        original_device = masked_features.device

        # Move to method device
        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)

        # ACO works on single instances, so we process batch one by one
        batch_size = masked_features.shape[0]
        selections = []

        for i in range(batch_size):
            x_obs = masked_features[i]
            obs_mask = feature_mask[i]

            # Get next feature from ACO oracle
            next_feature = self.aco_oracle.select_next_feature(
                x_obs, obs_mask, self.afa_classifier
            )

            if next_feature is not None:
                selections.append(next_feature)
            else:
                # If no beneficial acquisition, select any unobserved feature
                # or return -1 to indicate termination
                unobserved = (~obs_mask).nonzero(as_tuple=True)[0]
                if len(unobserved) > 0:
                    selections.append(unobserved[0].item())
                else:
                    selections.append(-1)  # No more features to acquire

        selection_tensor = torch.tensor(
            selections, dtype=torch.long).unsqueeze(-1)
        return selection_tensor.to(original_device)

    @override
    def predict(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Label:
        """
        Make prediction using the underlying classifier.
        """

        original_device = masked_features.device

        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)

        with torch.no_grad():
            prediction = self.afa_classifier(masked_features, feature_mask)

        return prediction.to(original_device)

    @override
    def save(self, path: Path):
        """
        Save the ACO method.
        """

        path.mkdir(exist_ok=True)

        # Save classifier
        self.afa_classifier.save(path / "classifier.pt")

        # Save ACO oracle parameters (we'll need to reconstruct it)
        oracle_params = {
            'k_neighbors': self.aco_oracle.density_estimator.k,
            'acquisition_cost': self.aco_oracle.acquisition_cost,
            'hide_val': self.aco_oracle.hide_val,
            'metric': self.aco_oracle.density_estimator.metric,
            'standardize': self.aco_oracle.density_estimator.standardize,
            'exhaustive_threshold': self.aco_oracle.subset_search.exhaustive_threshold,
            'max_samples': self.aco_oracle.subset_search.max_samples,
            'max_subset_size': self.aco_oracle.subset_search.max_subset_size,
        }
        torch.save(oracle_params, path / "oracle_params.pt")

        # Save training data for k-NN
        if self.aco_oracle.density_estimator.X_train is not None:
            torch.save({
                'X_train': self.aco_oracle.density_estimator.X_train,
                'y_train': self.aco_oracle.density_estimator.y_train,
                'scaler': self.aco_oracle.density_estimator.scaler,
            }, path / "training_data.pt")

        # Save classifier class name
        with open(path / "classifier_class_name.txt", "w") as f:
            f.write(self.afa_classifier.__class__.__name__)

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        """
        Load the ACO method.
        """

        from common.registry import get_afa_classifier_class

        # Load classifier
        with open(path / "classifier_class_name.txt") as f:
            classifier_class_name = f.read()
        afa_classifier = get_afa_classifier_class(classifier_class_name).load(
            path / "classifier.pt", device=device
        )

        # Load oracle parameters
        oracle_params = torch.load(
            path / "oracle_params.pt", map_location=device)

        # Reconstruct oracle
        aco_oracle = create_aco_oracle(**oracle_params)

        # Load training data and fit oracle
        training_data = torch.load(
            path / "training_data.pt", map_location=device)
        aco_oracle.density_estimator.X_train = training_data['X_train']
        aco_oracle.density_estimator.y_train = training_data['y_train']
        aco_oracle.density_estimator.scaler = training_data['scaler']

        # Refit k-NN (we need to reconstruct the sklearn part)
        aco_oracle.density_estimator.fit(
            training_data['X_train'], training_data['y_train']
        )

        return cls(
            aco_oracle=aco_oracle,
            afa_classifier=afa_classifier,
            _device=device
        )

    @override
    def to(self, device: torch.device) -> Self:
        """
        Move method to device.
        """

        self._device = device
        self.afa_classifier = self.afa_classifier.to(device)

        # Move training data if available
        if self.aco_oracle.density_estimator.X_train is not None:
            self.aco_oracle.density_estimator.X_train = \
                self.aco_oracle.density_estimator.X_train.to(device)
            self.aco_oracle.density_estimator.y_train = \
                self.aco_oracle.density_estimator.y_train.to(device)

        return self

    @property
    @override
    def device(self) -> torch.device:
        """
        Get device.
        """

        return self._device


class BehavioralCloningPolicy(nn.Module):
    """
    Neural network policy for behavioral cloning.
    """

    def __init__(
        self,
        n_features: int,
        num_cells: List[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        if num_cells is None:
            num_cells = [128, 128]

        self.n_features = n_features

        # Input: masked_features + feature_mask = 2 * n_features
        input_size = 2 * n_features

        layers = []
        prev_size = input_size

        for hidden_size in num_cells:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size

        # Output: probabilities over n_features + 1 (terminate action)
        layers.append(nn.Linear(prev_size, n_features + 1))

        self.network = nn.Sequential(*layers)

    def forward(self, masked_features: torch.Tensor, feature_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """

        # Concatenate masked features and mask
        x = torch.cat([masked_features, feature_mask.float()], dim=-1)
        return self.network(x)


@dataclass
@final
class ACOBCAFAMethod(AFAMethod):
    """
    Behavioral cloning version of ACO for faster inference.
    """

    policy_network: BehavioralCloningPolicy
    afa_classifier: AFAClassifier
    _device: torch.device = torch.device("cpu")

    def __post_init__(self):
        self.policy_network = self.policy_network.to(self._device)
        self.afa_classifier = self.afa_classifier.to(self._device)

    @override
    def select(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> AFASelection:
        """
        Select next feature using trained policy network.
        """

        original_device = masked_features.device

        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)

        with torch.no_grad():
            logits = self.policy_network(masked_features, feature_mask)

            # Mask out already observed features and invalid actions
            action_mask = ~feature_mask  # Can only select unobserved features
            logits[:, :-1] = logits[:, :-
                                    1].masked_fill(feature_mask, float('-inf'))

            # Select action with highest probability
            actions = logits.argmax(dim=-1)

        return actions.unsqueeze(-1).to(original_device)

    @override
    def predict(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Label:
        """
        Make prediction using the underlying classifier.
        """

        original_device = masked_features.device

        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)

        with torch.no_grad():
            prediction = self.afa_classifier(masked_features, feature_mask)

        return prediction.to(original_device)

    @override
    def save(self, path: Path):
        """
        Save the BC method.
        """

        path.mkdir(exist_ok=True)

        # Save policy network
        torch.save(self.policy_network.state_dict(), path / "policy.pt")

        # Save classifier
        self.afa_classifier.save(path / "classifier.pt")

        # Save network config
        config = {
            'n_features': self.policy_network.n_features,
            'num_cells': None,  # We'll need to extract this
            'dropout': None,    # We'll need to extract this
        }
        torch.save(config, path / "policy_config.pt")

        # Save classifier class name
        with open(path / "classifier_class_name.txt", "w") as f:
            f.write(self.afa_classifier.__class__.__name__)

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        """
        Load the BC method.
        """

        from common.registry import get_afa_classifier_class

        # Load classifier
        with open(path / "classifier_class_name.txt") as f:
            classifier_class_name = f.read()
        afa_classifier = get_afa_classifier_class(classifier_class_name).load(
            path / "classifier.pt", device=device
        )

        # Load policy config and network
        policy_config = torch.load(
            path / "policy_config.pt", map_location=device)
        policy_network = BehavioralCloningPolicy(**policy_config)
        policy_network.load_state_dict(
            torch.load(path / "policy.pt", map_location=device)
        )

        return cls(
            policy_network=policy_network,
            afa_classifier=afa_classifier,
            _device=device
        )

    @override
    def to(self, device: torch.device) -> Self:
        """
        Move method to device.
        """

        self._device = device
        self.policy_network = self.policy_network.to(device)
        self.afa_classifier = self.afa_classifier.to(device)
        return self

    @property
    @override
    def device(self) -> torch.device:
        """
        Get device.
        """

        return self._device


def generate_bc_training_data(
    aco_method: ACOAFAMethod,
    dataset: AFADataset,
    n_samples: int = 1000,
    max_steps: int = 20
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate training data for behavioral cloning by rolling out ACO policy.

    Returns:
        Tuple of (masked_features, feature_masks, actions)
    """

    features_list = []
    masks_list = []
    actions_list = []

    indices = torch.randperm(len(dataset))[:n_samples]

    logger.info(f"Generating BC training data from {n_samples} rollouts...")

    for i, idx in enumerate(indices):
        if i % 100 == 0:
            logger.info(f"Processed {i}/{n_samples} rollouts")

        feature, label = dataset[idx]

        # Start with no features observed
        current_mask = torch.zeros_like(feature, dtype=torch.bool)

        for step in range(max_steps):
            # Record current state
            masked_feature = feature.clone()
            masked_feature[~current_mask] = 0

            features_list.append(masked_feature)
            masks_list.append(current_mask.clone())

            # Get action from ACO
            action = aco_method.select(
                masked_feature.unsqueeze(0),
                current_mask.unsqueeze(0)
            )
            action_val = action.item()

            actions_list.append(action_val)

            # If terminate action or invalid action, break
            if action_val >= len(feature) or action_val < 0:
                break

            # Update mask
            current_mask[action_val] = True

            # If all features observed, break
            if current_mask.all():
                break

    return (
        torch.stack(features_list),
        torch.stack(masks_list),
        torch.tensor(actions_list, dtype=torch.long)
    )


def train_behavioral_cloning_policy(
    aco_method: ACOAFAMethod,
    dataset: AFADataset,
    n_features: int,
    bc_config,
    device: torch.device
) -> BehavioralCloningPolicy:
    """
    Train a behavioral cloning policy to imitate ACO.
    """

    # Generate training data
    X_features, X_masks, y_actions = generate_bc_training_data(
        aco_method, dataset, bc_config.bc_rollout_samples
    )

    # Create policy network
    policy = BehavioralCloningPolicy(
        n_features=n_features,
        num_cells=bc_config.bc_num_cells,
        dropout=bc_config.bc_dropout
    ).to(device)

    # Create data loader
    bc_dataset = TensorDataset(X_features, X_masks, y_actions)
    bc_loader = DataLoader(
        bc_dataset,
        batch_size=bc_config.bc_batch_size,
        shuffle=True
    )

    # Train policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=bc_config.bc_lr)
    loss_fn = nn.CrossEntropyLoss()

    policy.train()
    logger.info(f"Training BC policy for {bc_config.bc_epochs} epochs...")

    for epoch in range(bc_config.bc_epochs):
        total_loss = 0
        for batch_features, batch_masks, batch_actions in bc_loader:
            batch_features = batch_features.to(device)
            batch_masks = batch_masks.to(device)
            batch_actions = batch_actions.to(device)

            optimizer.zero_grad()

            logits = policy(batch_features, batch_masks)
            loss = loss_fn(logits, batch_actions)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            avg_loss = total_loss / len(bc_loader)
            logger.info(f"Epoch {epoch}: Average loss = {avg_loss:.4f}")

    policy.eval()
    return policy
