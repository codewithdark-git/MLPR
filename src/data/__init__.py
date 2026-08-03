"""Data module init file."""

from .dataset import MLPDataset
from .collator import MLPRTargetCollator

__all__ = ["MLPDataset", "MLPRTargetCollator"]
