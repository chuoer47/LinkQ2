"""Debug: numeric A/B of two-branch SDPA vs full causal SDPA."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

torch.manual_seed(0)
B, Hq, Hkv, Tq, Tk, D = 1, 16, 8, 2, 7, 128
q = torch.randn(B, Hq, Tq, D, device="cuda:0", dtype=torch.float16)
k = torch.randn(B, Hkv, Tk, D, device="cuda:0", dtype=torch.float16)
v = torch.randn(B, Hkv, Tk, D, device="cuda:0", dtype=torch.float16)

# full causal over all Tk
out_full = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

# two-branch: hist (5) full + new (2) causal
kv_start = Tk - Tq
out_hist = F.scaled_dot_product_attention(q, k[:, :, :kv_start], v[:, :, :kv_start],
                                          is_causal=False, enable_gqa=True)
out_new = F.scaled_dot_product_attention(q, k[:, :, kv_start:], v[:, :, kv_start:],
                                         is_causal=True, enable_gqa=True)
out_two = out_hist + out_new

print("full vs two-branch max diff:", (out_full - out_two).abs().max().item())
