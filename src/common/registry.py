from afa_rl.afa_methods import Shim2018AFAMethod
from common.datasets import CubeDataset

AFA_METHOD_REGISTRY = {"shim2018": Shim2018AFAMethod}

AFA_DATASET_REGISTRY = {"cube": CubeDataset}
