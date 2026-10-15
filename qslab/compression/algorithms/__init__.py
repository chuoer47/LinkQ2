"""Offline compression algorithms."""
from qslab.compression.algorithms.w4 import awq, awq_affine, rtn, rtn_clip

QUANTIZERS = {"rtn": rtn, "rtn_clip": rtn_clip, "awq": awq,
              "awq_affine": awq_affine}

__all__ = ["QUANTIZERS", "awq", "awq_affine", "rtn", "rtn_clip"]
