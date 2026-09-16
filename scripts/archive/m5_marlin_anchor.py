"""Marlin performance anchor: bench gptq_marlin_gemm on Qwen3-8B shapes.

Runs in the vllm env (torch 2.8 + prebuilt Marlin). Produces the performance
ceiling reference for our CUTLASS work. NOT part of the qslab engine.
"""
import torch
from vllm import _custom_ops as ops

SHAPES = [
    # (M, N, K) — N=out, K=in; marlin sig: gemm(a[M,K], b_q_weight[N,K/16 int32], scales, ...)
    (1, 4096, 4096),
    (1, 4096, 12288),
    (1, 12288, 4096),
    (8, 4096, 12288),
    (512, 4096, 12288),
    (4096, 4096, 12288),
]

def bench_marlin(M, N, K, iters=100):
    from vllm.scalar_type import scalar_types
    a = torch.randn(M, K, dtype=torch.float16, device="cuda")
    # marlin b: int32 packed [N, K/16] after repack; use repack from raw int4
    # raw gptq order: [K/8, N] int32 -> repack -> [N, K/16]
    b_raw = torch.randint(-2**31, 2**31 - 1, (K // 8, N), dtype=torch.int32, device="cuda")
    perm = torch.arange(K, dtype=torch.int, device="cuda")  # identity perm placeholder
    b_q = ops.gptq_marlin_repack(b_raw, perm, K, N, 4)
    # marlin scale repack: [N, K/g] fp16 -> permute via marlin_permute_scales
    scales = torch.randn(N, K // 128, dtype=torch.float16, device="cuda")
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_permute_scales
    scales = marlin_permute_scales(scales, K, N, 128)
    workspace = torch.zeros(N * 16, dtype=torch.int32, device="cuda")
    x = a
    for _ in range(10):
        y = ops.gptq_marlin_gemm(x, None, b_q, None, scales, None, None, None,
                                 None, workspace, scalar_types.uint4b8, M, N, K)
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(True); t1 = torch.cuda.Event(True)
    ts = []
    for _ in range(iters):
        t0.record()
        y = ops.gptq_marlin_gemm(x, None, b_q, None, scales, None, None, None,
                                 None, workspace, scalar_types.uint4b8, M, N, K)
        t1.record(); torch.cuda.synchronize()
        ts.append(t0.elapsed_time(t1))
    ts.sort()
    return ts[iters // 2]

for M, N, K in SHAPES:
    ms = bench_marlin(M, N, K)
    w_bytes = N * K / 2
    bw = (w_bytes + M * K * 2 + M * N * 2) / (ms / 1e3) / 1e9
    tflops = 2 * M * N * K / (ms / 1e3) / 1e12
    print(f"M={M:5d} N={N:5d} K={K:5d} | {ms*1e3:8.1f} us | {bw:7.1f} GB/s | {tflops:6.1f} TFLOPS")
