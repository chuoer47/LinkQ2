"""Marlin backend check: v1 pack -> marlin repack -> gemm vs dequant ref."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from quantizer.packfmt import pack_w4, unpack_w4
from kernels.qslab_kernels.marlin_backend import pack_v1_to_marlin, marlin_gemm
from kernels.qslab_kernels.marlin_ext import get_marlin

get_marlin()
torch.manual_seed(0)
O, I, g = 4096, 4096, 128
w = (torch.randn(O, I) * 0.02).to(torch.float16)
qfp, scale, zero = pack_w4(w, g)
B, s = pack_v1_to_marlin(qfp, scale, I, g)
print("marlin B:", tuple(B.shape), B.dtype, "| s:", tuple(s.shape), s.dtype)
x = (torch.randn(1, I) * 0.1).to(torch.float16).cuda()
B, s = B.cuda(), s.cuda()
workspace = torch.zeros(O // 128 * 16, dtype=torch.int32, device="cuda")
y_m = marlin_gemm(x, B, s, workspace)
y_ref = x @ unpack_w4(qfp, scale, zero, I, g).cuda().T
rel = ((y_m - y_ref).pow(2).sum() / y_ref.pow(2).sum()).item()
print(f"REL_ERR: {rel:.2e}")
