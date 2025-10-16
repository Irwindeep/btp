import torch

import dunesim.dataset as dataset
import dunesim.models as models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

__all__ = [
    "DEVICE",
    "dataset",
    "models",
]
