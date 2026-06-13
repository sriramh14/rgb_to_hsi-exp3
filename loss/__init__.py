"""Loss functions for RGB-to-HSI reconstruction."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mrae_loss import mrae_loss
from .rgb_consistency_loss import rgb_consistency_loss
from .sam_loss import sam_loss
from .spectral_gradient_loss import spectral_gradient_loss
from .spectral_smoothness_loss import spectral_smoothness_loss

__all__ = [
    "mrae_loss",
    "rgb_consistency_loss",
    "sam_loss",
    "spectral_gradient_loss",
    "spectral_smoothness_loss",
]
