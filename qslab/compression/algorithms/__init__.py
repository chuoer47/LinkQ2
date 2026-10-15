"""Offline compression algorithms."""
from qslab.quantization import QuantizationMethod
from .base import WeightQuantizer
from .rtn import RTN
from .rtn_clip import RTNClip
from .awq import AWQ
from .awq_affine import AWQAffine

QUANTIZERS = {QuantizationMethod.RTN: RTN, QuantizationMethod.RTN_CLIP: RTNClip,
              QuantizationMethod.AWQ: AWQ, QuantizationMethod.AWQ_AFFINE: AWQAffine}

__all__ = ["WeightQuantizer", "QUANTIZERS", "RTN", "RTNClip", "AWQ", "AWQAffine"]
