"""Algorithm identifiers shared by checkpoint producers and consumers."""
from enum import Enum


class QuantizationMethod(str, Enum):
    RTN = 'rtn_w4a16'
    RTN_CLIP = 'rtn_clip_w4a16'
    AWQ = 'awq_w4a16'
    AWQ_AFFINE = 'awq_affine_w4a16'

    def __str__(self):
        return self.value
