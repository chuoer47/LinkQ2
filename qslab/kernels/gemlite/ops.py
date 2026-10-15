"""Thin launch interface for the unmodified GemLite 0.4.8 W4 kernels."""
from .triton_kernels.gemv_revsplitK_A16fWnO16f_int32packing import gemv_revsplitK_A16fWnO16f
from .triton_kernels.gemm_splitK_A16fWnO16f_int32packing import gemm_splitK_A16fWnO16f
from .triton_kernels.gemm_A16fWnO16f_int32packing import gemm_A16fWnO16f

ROUTES = {"GEMV_REVSPLITK": gemv_revsplitK_A16fWnO16f,
          "GEMM_SPLITK": gemm_splitK_A16fWnO16f, "GEMM": gemm_A16fWnO16f}


def linear(x, packed, scales, zeros, meta):
    shape = x.shape[:-1] + (packed.shape[1],)
    x = x.contiguous().view(-1, x.shape[-1])
    m = x.shape[0]
    route = "GEMV_REVSPLITK" if m == 1 else ("GEMM_SPLITK" if m <= 64 else "GEMM")
    return ROUTES[route].forward(x, packed, scales, zeros, None, *meta).view(shape)
