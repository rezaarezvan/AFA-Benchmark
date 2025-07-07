import timeit

import torch

# Setup for the test
masked_features = torch.randn(64, 128)  # Adjust size as needed
feature_mask = torch.randint(0, 2, (64, 128))  # Adjust size as needed

# Timer for `get_feature_set`
time_get_feature_set_old = timeit.timeit(
    "get_feature_set_old(masked_features, feature_mask)",
    setup="from afa_rl.utils import get_feature_set_old\nfrom __main__ import masked_features, feature_mask",
    number=100,  # Number of repetitions for the test
)

# Timer for `get_feature_set_vec`
time_get_feature_set = timeit.timeit(
    "get_feature_set(masked_features, feature_mask)",
    setup="from afa_rl.utils import get_feature_set\nfrom __main__ import masked_features, feature_mask",
    number=100,  # Number of repetitions for the test
)

print(f"Time for `get_feature_set`: {time_get_feature_set:.6f} seconds")
