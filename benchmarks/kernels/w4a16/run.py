from benchmarks.common import publish
"""Packed W4A16 versus dense FP16/Fake Linear, optionally original GemLite."""
import argparse
import gc
import importlib
import json
import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.benchmark import Timer
from triton.testing import do_bench_cudagraph

from qslab.runtime.backends.gemlite import GemLiteLinear
from qslab.kernels.gemlite.core import load_config, save_config


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--out', type=Path, default=Path('results/kernels/w4a16'))
    parser.add_argument('--reference', type=Path, help='解压后的 GemLite 0.4.8 源码目录；仅作对照')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.out.mkdir(parents=True, exist_ok=True)
    cache = Path('.cache/benchmarks/gemlite.json')
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        load_config(cache)
    original = None
    if args.reference:
        sys.path.insert(0, str(args.reference))
        original = importlib.import_module('gemlite')
        original.reset_config()
    torch.manual_seed(2026)
    torch.set_num_threads(4)
    rows = []
    for symmetric, group, n, k in [(True,128,4096,4096), (True,128,12288,4096),
                                  (True,128,4096,12288), (False,64,1024,4096)]:
        codes = torch.randint(-8 if symmetric else 0, 8 if symmetric else 16, (n,k), dtype=torch.int8)
        scale = (torch.rand(n,k//group)*0.01+0.001).half()
        zero = None if symmetric else torch.randint(0,16,scale.shape).half()
        layer = GemLiteLinear.from_quantized(codes,scale,group_size=group,symmetric=symmetric,
                                            zero_point=zero,device='cuda',dtype=torch.float16)
        dense = ((codes.float().reshape(n,k//group,group) -
                  (0 if symmetric else zero.float().unsqueeze(-1))) * scale.float().unsqueeze(-1)).reshape(n,k).half().cuda()
        native = torch.randn(n,k,device='cuda',dtype=torch.float16)*0.05
        upstream = None
        if original:
            upstream = original.GemLiteLinear(4,group,k,n,input_dtype=original.DType.FP16,
                output_dtype=original.DType.FP16,acc_dtype=original.DType.FP32).pack(
                    (codes.to('cuda',dtype=torch.int16)+(8 if symmetric else 0)).to(torch.uint8),
                    scale.cuda(),8 if symmetric else zero.cuda(),fma_mode=False)
            assert all(torch.equal(a,b) for a,b in [(layer.packed,upstream.W_q),
                       (layer.scales,upstream.scales),(layer.zeros,upstream.zeros)])
        for m in [1,32,128,512]:
            x = torch.randn(m,k,device='cuda',dtype=torch.float16)*0.1
            functions = {'fp16': lambda:F.linear(x,native), 'fake':lambda:F.linear(x,dense), 'real':lambda:layer(x)}
            if upstream:
                functions['upstream'] = lambda:upstream(x)
            actual = layer(x).float()
            expected = F.linear(x,dense).float()
            row = {'m':m,'n':n,'k':k,'group_size':group,'symmetric':symmetric,
                   'relative_l2_vs_fake':float((actual-expected).norm()/expected.norm())}
            assert row['relative_l2_vs_fake'] < 0.01
            if upstream:
                reference = upstream(x).float()
                row['packed_equal'] = True
                row['relative_l2_vs_upstream'] = float((actual-reference).norm()/reference.norm())
                assert row['relative_l2_vs_upstream'] < 0.005
            for name, fn in functions.items():
                fn()
                row[name+'_gpu_ms'] = do_bench_cudagraph(fn,rep=40)
                row[name+'_host_ms'] = Timer('fn(); torch.cuda.synchronize()',
                    globals={'fn':fn,'torch':torch}).blocked_autorange(min_run_time=0.1).median*1000
            row['gpu_speedup_vs_fp16'] = row['fp16_gpu_ms']/row['real_gpu_ms']
            row={k:round(v,6) if isinstance(v,float) else v for k,v in row.items()}
            rows.append(row)
            print(json.dumps(row),flush=True)
        del layer,dense,native,upstream,x,actual,expected,functions,fn
        gc.collect()
        torch.cuda.empty_cache()
    save_config(cache)
    from benchmarks.common import save, table
    save(args.out/'results.json',{'measurements':rows})
    table(args.out/'report.md','W4A16 单算子延迟',['M','N','K','FP16 GPU μs','W4 GPU μs','加速比','相对 L2'],
          [[r['m'],r['n'],r['k'],round(r['fp16_gpu_ms']*1000,2),round(r['real_gpu_ms']*1000,2),
            round(r['gpu_speedup_vs_fp16'],3),round(r['relative_l2_vs_fake'],6)] for r in rows],
          'GPU 时间由 Triton CUDA Graph 计时工具测量；JSON 另保留同步主机调用时间。单算子加速比不等于整模加速比。')

    publish(args.out,Path(__file__).parent)


if __name__ == '__main__':
    main()
