"""Deterministic supported shape matrix for quantized linear kernels."""
from __future__ import annotations

CASES = (
    {"name": "decode_small", "m": 1, "n": 512, "k": 1024},
    {"name": "decode_batch", "m": 4, "n": 512, "k": 1024},
    {"name": "prefill_batch16", "m": 16, "n": 512, "k": 1024},
    {"name": "prefill_medium", "m": 32, "n": 1024, "k": 2048},
)

BACKENDS = ("qslab_w4a16", "marlin", "nunchaku_awq")
ALGORITHMS = {"qslab_w4a16": "awq", "marlin": "awq",
              "nunchaku_awq": "nunchaku_awq"}
GROUP_SIZES = {"qslab_w4a16": 128, "marlin": 128,
               "nunchaku_awq": 64}


def supports(backend: str, case: dict) -> tuple[bool, str | None]:
    m, n, k = case["m"], case["n"], case["k"]
    if backend in ("qslab_w4a16", "marlin") and (k % 128 or n % 256):
        return False, "requires K divisible by 128 and N divisible by 256"
    if backend == "nunchaku_awq":
        if k % 64 or n % 16:
            return False, "AWQ GEMV requires K divisible by 64 and N divisible by 16"
        if m <= 0:
            return False, "M must be positive"
    return True, None
