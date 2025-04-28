def set_seed(seed: int):
    import os
    import random
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from jaxtyping import Float, Bool
from torch import Tensor


def get_class_probabilities(labels: Bool[Tensor, "*batch n_classes"]) -> Float[Tensor, "n_classes"]:
    """
    Returns the class probabilities for a given set of labels.
    """
    class_counts = labels.float().sum(dim=0)
    class_probabilities = class_counts / class_counts.sum()
    return class_probabilities
