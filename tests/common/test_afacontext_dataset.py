import pytest
from common.datasets import AFAContextDataset


@pytest.fixture
def setup_data():
    data = AFAContextDataset(n_samples=1000, std_bin=0.0)
    return data


def bin_list_to_int(bin_list: list[int]):
    """Convert a list of binary values to an integer."""
    return int("".join(str(x) for x in bin_list), 2)


def test_afa(setup_data):
    for features, label in setup_data:
        cls_idx = label.argmax(-1).item()
        context = features[0].int().item()
        range_start = context * 3 + 1
        range_end = context * 3 + 4
        # assert bin_list_to_int(features[range_start:range_end].int().tolist()) == cls_idx
        # Reversed apparently
        assert (
            bin_list_to_int(reversed(features[range_start:range_end].int().tolist()))
            == cls_idx
        )
