import torch

from torch import nn
from torch.utils.data import TensorDataset
from torch.distributions import RelaxedOneHotCategorical

from afabench.common.custom_types import AFADataset


def transform_dataset(dataset: AFADataset, selected_features):
    x = dataset.features
    y = dataset.labels
    x_selected = x[:, selected_features]
    return TensorDataset(x_selected, y)


def restore_parameters(model, best_model):
    """Move parameters from best model to current model."""
    for param, best_param in zip(
        model.parameters(), best_model.parameters(), strict=False
    ):
        param.data = best_param


class ConcreteMask(nn.Module):
    """
    For differentiable global feature selection.

    Args:
      num_features:
      num_select:
      group_matrix:
      append:
      gamma:

    """

    def __init__(
        self,
        num_features,
        num_select,
        group_matrix=None,
        append=False,
        gamma=0.2,
    ):
        super().__init__()
        self.logits = nn.Parameter(
            torch.randn(num_select, num_features, dtype=torch.float32)
        )
        self.append = append
        self.gamma = gamma
        if group_matrix is None:
            self.group_matrix = None
        else:
            self.register_buffer("group_matrix", group_matrix.float())

    def forward(self, x, temp):
        dist = RelaxedOneHotCategorical(temp, logits=self.logits / self.gamma)
        sample = dist.rsample([len(x)])
        m = sample.max(dim=1).values
        if self.group_matrix is not None:
            out = x * (m @ self.group_matrix)
        else:
            out = x * m
        if self.append:
            out = torch.cat([out, m], dim=1)
        return out
