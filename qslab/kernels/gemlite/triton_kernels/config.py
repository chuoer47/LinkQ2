"""Configuration needed by the three retained upstream kernels."""


class AUTOTUNE:
    GEMV_REVSPLITK = "fast"
    GEMM_SPLITK = "fast"
    GEMM = "fast"
    USE_CUDA_GRAPH = False


class KERNEL:
    ENABLE_CACHING = False
    CACHE_SIZE = 256
