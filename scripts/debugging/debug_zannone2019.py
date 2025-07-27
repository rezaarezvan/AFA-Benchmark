import torch
import wandb
from afa_rl.zannone2019.reward import get_zannone2019_reward_fn
from afa_rl.zannone2019.utils import load_pretrained_model_artifacts
from common.custom_types import FeatureMask, Label, MaskedFeatures
from torch.nn import functional as F

import random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

# random.seed(0)
# np.random.seed(0)
# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

run = wandb.init()


(
    _train_dataset,
    _val_dataset,
    _test_dataset,
    _dataset_metadata,
    model,
    pretrain_config,
) = load_pretrained_model_artifacts("pretrain_zannone2019-cubeSimple_split_1:v0")

model.eval()

reward_fn = get_zannone2019_reward_fn(model, weights=torch.ones(8))


def reward_and_predictions(
    masked_features: MaskedFeatures, feature_mask: FeatureMask, label: Label
) -> tuple[torch.Tensor, torch.Tensor]:
    _encoding, mu, _logvar, z = model.partial_vae.encode(masked_features, feature_mask)
    logits = model.classifier(mu)
    reward = -F.cross_entropy(logits, label)
    predictions = logits.softmax(dim=-1)
    return reward, predictions


def plot_reward(features, feature_mask, label):
    matplotlib.use("WebAgg")

    num_runs = 10
    # feature_indices = [i for i in range(0, 10) if i != 3]
    feature_indices = (~feature_mask).nonzero()[:, -1]
    rewards_matrix = np.zeros((num_runs, len(feature_indices)))

    for run_idx in range(num_runs):
        for feat_idx, next_feature_idx in enumerate(feature_indices):
            new_feature_mask = (
                feature_mask
                + F.one_hot(
                    torch.tensor([next_feature_idx]), num_classes=features.shape[-1]
                )
            ).bool()
            new_masked_features = features * new_feature_mask
            # reward, preds = reward_and_predictions(
            #     new_masked_features, new_feature_mask, label
            # )
            batch_size = features.shape[0]
            reward = reward_fn(
                _masked_features=features,  # not used,
                _feature_mask=feature_mask,
                new_masked_features=new_masked_features,
                new_feature_mask=new_feature_mask,
                _afa_selection=torch.full((batch_size, 1), torch.nan),  # not used
                _features=features,
                label=label,
                _done=torch.full((batch_size, 1), torch.nan),  # not used
            )
            rewards_matrix[run_idx, feat_idx] = reward.item()

    mean_rewards = rewards_matrix.mean(axis=0)
    std_rewards = rewards_matrix.std(axis=0)

    plt.errorbar(feature_indices, mean_rewards, yerr=std_rewards, fmt="-o")
    plt.xticks(range(features.shape[-1]))
    plt.xlabel("Feature Index")
    plt.ylabel("Reward")
    plt.title("Reward for Next Feature Choice (mean ± std over 10 runs)")
    plt.show()


def plot_predictions():
    matplotlib.use("WebAgg")

    features = torch.tensor([1, 0.5, 1, 0.5, 0, 0.5]).unsqueeze(0)  # class 3
    # masked_features = torch.tensor([0, 0, 0, 1, 0, 0]).unsqueeze(0)
    masked_features = features
    feature_mask = torch.tensor(
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.bool
    ).unsqueeze(0)
    label = torch.tensor([0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.float32).unsqueeze(
        0
    )  # index 3
    # true features for this is [0 0 0 [1 1 0] 0 0 0 0]

    feature_indices = [i for i in range(0, 10) if i != 3]
    all_predictions = []

    for feat_idx, next_feature_idx in enumerate(feature_indices):
        new_feature_mask = (
            feature_mask + F.one_hot(torch.tensor([next_feature_idx]), num_classes=10)
        ).bool()
        new_masked_features = features * new_feature_mask
        reward, preds = reward_and_predictions(
            new_masked_features, new_feature_mask, label
        )
        all_predictions.append(preds[0].detach().cpu().numpy())

    fig, axs = plt.subplots(9)
    for i, (feature_idx, data) in enumerate(zip(feature_indices, all_predictions)):
        # data shape: (num_runs, num_classes)
        axs[i].bar(torch.arange(8), data)
        axs[i].set_title(f"selecting feature {feature_idx}")

    plt.show()


features = torch.tensor([1, 0.5, 1, 0.5, 0, 0.5]).unsqueeze(0)  # class 3
feature_mask = torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.bool).unsqueeze(0)
label = torch.tensor([0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.float32).unsqueeze(
    0
)  # index 3

plot_reward(features, feature_mask, label)
