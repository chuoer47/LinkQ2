"""Compare official packing and numerical error on actual Qwen3 projection shapes."""
import argparse
import gc
import json
from pathlib import Path

import marlin
import torch
from torch import nn
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph

from qslab.runtime.backends.marlin import MarlinLinear


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(2026)
    torch.set_num_threads(4)
    rows = []
    for n, k in [(4096, 4096), (1024, 4096), (12288, 4096), (4096, 12288)]:
        codes = torch.randint(-8, 8, (n, k), dtype=torch.int8, device='cuda')
        scales = (torch.rand(n, k // 128, device='cuda') * 0.01 + 0.001).half()
        dense = (codes.float().reshape(n, -1, 128) * scales.float().unsqueeze(-1)).reshape(n, k).half()
        layer = MarlinLinear.from_quantized(codes, scales, group_size=128, symmetric=True,
                                           device='cuda', dtype=torch.float16)
        linear = nn.Linear(k, n, bias=False, device='meta', dtype=torch.float16)
        linear.weight = nn.Parameter(dense, requires_grad=False)
        official = marlin.Layer(k, n, groupsize=128).cuda()
        official.pack(linear, scales)
        assert torch.equal(layer.operator.B, official.B), 'packed weights differ from official pack'
        assert torch.equal(layer.operator.s, official.s), 'scales differ from official pack'
        for m in [1, 16, 128, 512]:
            x = torch.randn(m, k, device='cuda', dtype=torch.float16) * 0.1
            actual = layer(x).float()
            expected = F.linear(x, dense).float()
            delta = actual - expected
            row = dict(m=m, n=n, k=k, packed_equal=True, scales_equal=True,
                       mean_abs_relative=float(delta.abs().mean() / expected.abs().mean()),
                       relative_l2=float(delta.norm() / expected.norm()), max_abs=float(delta.abs().max()))
            rows.append(row)
            args.out.write_text(json.dumps({'cases': rows}, indent=2))
            assert torch.isfinite(actual).all() and row['mean_abs_relative'] < 0.001, row
            row['fp16_gpu_ms'] = do_bench_cudagraph(lambda: F.linear(x, dense), rep=40)
            row['marlin_gpu_ms'] = do_bench_cudagraph(lambda: layer(x), rep=40)
            row['gpu_speedup_vs_fp16'] = row['fp16_gpu_ms'] / row['marlin_gpu_ms']
            print(json.dumps(row), flush=True)
        del codes, scales, dense, linear, official, layer, x, actual, expected, delta
        gc.collect()
        torch.cuda.empty_cache()
    args.out.write_text(json.dumps({'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
        'gpu_timing': 'CUDA graph replay, hot weights; compilation and packing excluded', 'cases': rows}, indent=2))


if __name__ == '__main__':
    main()
