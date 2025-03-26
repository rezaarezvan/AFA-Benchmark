import torch

from afa_rl.utils import get_feature_set


def test_observed_features_to_state():
    # First case
    x1 = torch.tensor([3, 0, 2], dtype=torch.float32)
    z1 = torch.tensor([1, 0, 1], dtype=torch.bool)
    s1 = torch.tensor([[3, 1, 0, 0], [0, 0, 0, 0], [2, 0, 0, 1]], dtype=torch.float32)

    # Second case
    x2 = torch.tensor([5, 0, 0], dtype=torch.float32)
    z2 = torch.tensor([1, 1, 0], dtype=torch.bool)
    s2 = torch.tensor([[5, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=torch.float32)

    # Batched, stack them
    x = torch.stack([x1, x2])
    z = torch.stack([z1, z2])
    s = torch.stack([s1, s2])

    s_hat = get_feature_set(x, z)
    assert torch.allclose(s_hat, s)
