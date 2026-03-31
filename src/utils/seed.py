"""Random seed utilities for reproducibility"""

import random
from typing import Optional

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional in lightweight envs
    torch = None


def set_seed(seed: Optional[int] = None) -> int:
    """
    Set random seed for reproducibility.

    Falls back to Python and NumPy only when torch is unavailable.
    """
    if seed is None:
        seed = 42

    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    return seed
