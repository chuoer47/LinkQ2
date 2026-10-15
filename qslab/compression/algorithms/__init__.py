"""Offline compression algorithms."""
from .base import WeightQuantizer
from .rtn import RTN
from .rtn_clip import RTNClip
from .awq import AWQ
from .awq_affine import AWQAffine

QUANTIZERS = {"rtn": RTN, "rtn_clip": RTNClip, "awq": AWQ,
              "awq_affine": AWQAffine}

__all__ = ["WeightQuantizer", "QUANTIZERS", "RTN", "RTNClip", "AWQ", "AWQAffine"]
