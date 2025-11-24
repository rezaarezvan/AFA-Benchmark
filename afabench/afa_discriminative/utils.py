import torch
from torch import nn
from torch.distributions import Categorical, RelaxedOneHotCategorical


def restore_parameters(model, best_model):
    """Move parameters from best model to current model."""
    for param, best_param in zip(
        model.parameters(), best_model.parameters(), strict=False
    ):
        param.data = best_param


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


class MaskLayer2d(nn.Module):
    """
    Mask layer for zeroing out 2d image data.

    Args:
      mask_width: width of the mask, or the number of patches.
      patch_size: upsampling factor for the mask, or number of pixels along
        the side of each patch.
      append: whether to append mask to the output.
    """

    def __init__(self, mask_width, patch_size, append=False):
        super().__init__()
        self.append = append
        self.mask_width = mask_width
        self.mask_size = mask_width**2

        # Set up upsampling.
        self.patch_size = patch_size
        if patch_size == 1:
            self.upsample = nn.Identity()
        elif patch_size > 1:
            self.upsample = nn.Upsample(scale_factor=patch_size)
        else:
            raise ValueError("patch_size should be int >= 1")

    def forward(self, x, mask):
        # Reshape if necessary.
        if len(mask.shape) == 2:
            mask = mask.reshape(-1, 1, self.mask_width, self.mask_width)
        elif len(mask.shape) != 4:
            raise ValueError(
                f"cannot determine how to reshape mask with shape = {
                    mask.shape
                }"
            )

        # Apply mask.
        mask = self.upsample(mask)
        out = x * mask
        if self.append:
            out = torch.cat([out, mask], dim=1)
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
        dist = RelaxedOneHotCategorical(temp, logits=logits / self.gamma)
        return dist.rsample()
