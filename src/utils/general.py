import random
import numpy as np
import torch


def set_seed_everywhere(seed: int) -> None:
    # Set torch and numpy and random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
