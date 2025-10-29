"""Random seed utilities for reproducibility"""

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: Optional[int] = None) -> int:
    """
    Set random seed for reproducibility

    Args:
        seed: Random seed value. If None, will use 42.

    Returns:
        The seed value that was set
    """
    if seed is None:
        seed = 42

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed
