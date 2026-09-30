"""Offline compression algorithms."""
from qslab.compression.algorithms.w4 import awq, nunchaku_awq, rtn, rtn_clip

QUANTIZERS = {"rtn": rtn, "rtn_clip": rtn_clip, "awq": awq,
              "nunchaku_awq": nunchaku_awq}

__all__ = ["QUANTIZERS", "awq", "nunchaku_awq", "rtn", "rtn_clip"]
