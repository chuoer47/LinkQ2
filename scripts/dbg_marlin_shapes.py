import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from quantizer.packfmt import pack_w4
from kernels.qslab_kernels.marlin_backend import pack_v1_to_marlin, marlin_gemm
from kernels.qslab_kernels.marlin_ext import get_marlin
get_marlin()
G = 128
torch.manual_seed(0)
N, K = 4096, 4096
w = (torch.randn(N, K) * 0.02).to(torch.float16)
qfp, scale, zero = pack_w4(w, G)
B, s = pack_v1_to_marlin(qfp, scale, K, G)
B, s = B.cuda(), s.cuda()
ws = torch.zeros(N // 128 * 16, dtype=torch.int32, device="cuda")
for M in [1, 2, 3, 5, 7, 8, 9, 17, 33, 64, 512]:
    x = torch.randn(M, K, dtype=torch.float16, device="cuda")
    try:
        y = marlin_gemm(x, B, s, ws)
        print(f"M={M}: OK", flush=True)
    except Exception as e:
        print(f"M={M}: FAIL {str(e)[:80]}", flush=True)
