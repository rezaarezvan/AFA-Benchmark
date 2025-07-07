import torch
import torch.nn as nn
import numpy as np
from torch.distributions import RelaxedOneHotCategorical, Categorical


def restore_parameters(model, best_model):
    """Move parameters from best model to current model."""
    for param, best_param in zip(model.parameters(), best_model.parameters()):
        param.data = best_param


def generate_uniform_mask(batch_size, num_features):
    """Generate binary masks with cardinality chosen uniformly at random."""
    unif = torch.rand(batch_size, num_features)
    ref = torch.rand(batch_size, 1)
    return (unif > ref).float()


def make_onehot(x):
    """Make an approximately one-hot vector one-hot."""
    argmax = torch.argmax(x, dim=1)
    onehot = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
    onehot[torch.arange(len(x)), argmax] = 1
    return onehot


def get_entropy(pred):
    """Calculate entropy, assuming logit predictions."""
    return Categorical(logits=pred).entropy()


def ind_to_onehot(inds, n):
    """Convert index to one-hot encoding."""
    onehot = torch.zeros(len(inds), n, dtype=torch.float32, device=inds.device)
    onehot[torch.arange(len(inds)), inds] = 1
    return onehot


class MaskLayer(nn.Module):
    """
    Mask layer for tabular data.

    Args:
      append:
      mask_size:
    """

    def __init__(self, append, mask_size=None):
        super().__init__()
        self.append = append
        self.mask_size = mask_size

    def forward(self, x, m):
        out = x * m
        if self.append:
            out = torch.cat([out, m], dim=1)
        return out


class ConcreteSelector(nn.Module):
    """Output layer for selector models."""

    def __init__(self, gamma=0.2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, temp, deterministic=False):
        if deterministic:
            # TODO this is somewhat untested, but seems like best way to preserve argmax
            return torch.softmax(logits / (self.gamma * temp), dim=-1)
        else:
            dist = RelaxedOneHotCategorical(temp, logits=logits / self.gamma)
            return dist.rsample()
