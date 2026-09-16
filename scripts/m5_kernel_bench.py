"""M5: three-way kernel benchmark — v1 custom / marlin / cublas fp16.

Real Qwen3-8B shapes, M in {1, 8, 512, 4096}. L2 flush for M<=8 (decode).
Saves results/m5_kernel_bench.json.
"""
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from qslab.quant.packfmt import pack_w4, unpack_w4
from qslab.kernels.marlin_backend import pack_v1_to_marlin, marlin_gemm
from qslab.kernels.marlin_ext import get_marlin
from qslab.kernels.ops import w4a16_gemm

get_marlin()

SHAPES = [  # (N=out, K=in) — Qwen3-8B linears
    (4096, 4096),
    (4096, 12288),
    (12288, 4096),
]
MS = [1, 8, 512, 4096]
G = 128


def flush_l2():
    # 96MB dummy buffer write to evict L2 (72MB on 4090)
    global _flush_buf
    try:
        _flush_buf
    except NameError:
        _flush_buf = torch.empty(96 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    _flush_buf.fill_(1.0)


def bench(fn, iters=200, flush=False):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(True)
    t1 = torch.cuda.Event(True)
    ts = []
    for i in range(iters):
        if flush:
            flush_l2()
        t0.record()
        fn()
        t1.record()
        torch.cuda.synchronize()
        ts.append(t0.elapsed_time(t1))
    return statistics.median(ts)


results = {}
for N, K in SHAPES:
    torch.manual_seed(0)
    w = (torch.randn(N, K) * 0.02).to(torch.float16).cuda()
    qfp, scale, zero = pack_w4(w, G)
    B, s = pack_v1_to_marlin(qfp, scale, K, G)
    qfp, scale = qfp.cuda(), scale.cuda()
    B, s = B.cuda(), s.cuda()
    zero_gpu = zero.cuda()
    w_deq = unpack_w4(qfp, scale, zero_gpu, K, G).cuda()
    workspace = torch.zeros(N // 128 * 16, dtype=torch.int32, device="cuda")
    for M in MS:
        x = torch.randn(M, K, dtype=torch.float16, device="cuda")
        flush = M <= 8
        t_v1 = bench(lambda: w4a16_gemm(qfp, scale, x, G), flush=flush)
        t_mar = bench(lambda: marlin_gemm(x, B, s, workspace), flush=flush)
        t_cb = bench(lambda: x @ w_deq.T, flush=flush)
        results[f"N{N}_K{K}_M{M}"] = {
            "v1_us": round(t_v1 * 1e3, 1),
            "marlin_us": round(t_mar * 1e3, 1),
            "cublas_us": round(t_cb * 1e3, 1),
            "v1_bw_gbs": round((N * K / 2 + M * (K + N) * 2) / t_v1 / 1e6, 0),
            "marlin_bw_gbs": round((N * K / 2 + M * (K + N) * 2) / t_mar / 1e6, 0),
            "cublas_bw_gbs": round((N * K * 2 + M * (K + N) * 2) / t_cb / 1e6, 0),
        }
        r = results[f"N{N}_K{K}_M{M}"]
        print(f"N{N:5d} K{K:5d} M{M:4d} | v1 {r['v1_us']:8.1f}us ({r['v1_bw_gbs']:6.0f} GB/s) | "
              f"marlin {r['marlin_us']:8.1f}us ({r['marlin_bw_gbs']:6.0f} GB/s) | "
              f"cublas {r['cublas_us']:8.1f}us ({r['cublas_bw_gbs']:6.0f} GB/s)", flush=True)

results["timestamp"] = datetime.now().isoformat()
Path("results/m5_kernel_bench.json").write_text(json.dumps(results, indent=2))
print("saved results/m5_kernel_bench.json")
