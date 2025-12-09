from typing import override

import torch
from torch import nn
from torch.distributions import Categorical, RelaxedOneHotCategorical


def restore_parameters(model: nn.Module, best_model: nn.Module) -> None:
    """Move parameters from best model to current model."""
    for param, best_param in zip(
        model.parameters(), best_model.parameters(), strict=False
    ):
        param.data = best_param


def make_onehot(x: torch.Tensor) -> torch.Tensor:
    """Make an approximately one-hot vector one-hot."""
    argmax = torch.argmax(x, dim=1)
    onehot = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
    onehot[torch.arange(len(x)), argmax] = 1
    return onehot


def get_entropy(pred: torch.Tensor) -> torch.Tensor:
    """Calculate entropy, assuming logit predictions."""
    return Categorical(logits=pred).entropy()


def ind_to_onehot(inds: torch.Tensor, n: int) -> torch.Tensor:
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

    def __init__(self, append: bool, mask_size: int | None = None):
        super().__init__()
        self.append: bool = append
        self.mask_size: int | None = mask_size

    @override
    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        out = x * m
        if self.append:
            out: torch.Tensor = torch.cat([out, m], dim=1)
        return out


class MaskLayer2d(nn.Module):
    """
    Mask layer for zeroing out 2d image data.

    Args:
      mask_width: width of the mask, or the number of patches.
      patch_size: upsampling factor for the mask, or number of pixels along
        the side of each patch.
      append: whether to append mask to the output.
    """

    def __init__(self, mask_width: int, patch_size: int, append: bool):
        super().__init__()
        self.append: bool = append
        self.mask_width: int = mask_width
        self.mask_size: int = mask_width**2

        # Set up upsampling.
        self.patch_size: int = patch_size
        self.upsample: nn.Module
        if patch_size == 1:
            self.upsample = nn.Identity()
        elif patch_size > 1:
            self.upsample = nn.Upsample(scale_factor=patch_size)
        else:
            msg = "patch_size should be int >= 1"
            raise ValueError(msg)

    @override
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Reshape if necessary.
        if len(mask.shape) == 2:
            mask = mask.reshape(-1, 1, self.mask_width, self.mask_width)
        elif len(mask.shape) != 4:
            msg = f"cannot determine how to reshape mask with shape = {
                mask.shape
            }"
            raise ValueError(msg)

        # Apply mask.
        mask = self.upsample(mask)
        out = x * mask
        if self.append:
            out = torch.cat([out, mask], dim=1)
        return out


class ConcreteSelector(nn.Module):
    """Output layer for selector models."""

    def __init__(self, gamma: float = 0.2) -> None:
        super().__init__()
        self.gamma: float = gamma

    @override
    def forward(
        self,
        logits: torch.Tensor,
        temp: float,
        deterministic: bool = False,
    ) -> torch.Tensor:
        if deterministic:
            return torch.softmax(logits / (self.gamma * temp), dim=-1)
        dist = RelaxedOneHotCategorical(temp, logits=logits / self.gamma)
        return dist.rsample()
