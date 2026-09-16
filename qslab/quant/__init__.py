"""L1 quantization layer: algorithms, packing formats, calibration, KV plans."""
from qslab.quant.packfmt import pack_w4, unpack_w4, save_qslab_w4, load_qslab_w4  # noqa: F401
from qslab.quant.w4 import quantize_weight, rtn_quantize_weight, awq_find_scales  # noqa: F401
from qslab.quant.calibrate import collect_activations  # noqa: F401

# Import the backend implementations so their @register decorators run.
# (Without this, a bare ``import qslab.quant`` leaves QUANT_BACKENDS empty.)
from qslab.quant import w4_backends  # noqa: E402,F401
