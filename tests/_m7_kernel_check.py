"""M7-S1: paged KV4 correctness — paged kernel vs a dense reference.

Builds a synthetic K/V sequence, writes it through the paged cache, and
compares the paged attention output against a straightforward dense
reference computed on the SAME quantized values (so the only difference is
summation order / block boundaries, not quantization error).
"""
import sys
sys.path.insert(0, "/home/<user>/<workdir>/qserve-lab")

import torch

from qslab.quant.cache.kv4_paged import KV4PagedCache

torch.manual_seed(0)
H, D, T, G = 8, 128, 300, 64          # 300 tokens -> spans 3 blocks (128 each)
DEV = "cuda:0"

k_all = (torch.randn(1, H, T, D) * 0.5).to(torch.float16).to(DEV)
v_all = (torch.randn(1, H, T, D) * 0.5).to(torch.float16).to(DEV)

cache = KV4PagedCache(batch=1, num_kv_heads=H, head_dim=D, max_len=1024,
                      device=DEV, group=G, block_n=128)

# append in decode-like increments (1, then 63, then 1, then 235)
for lo, hi in [(0, 1), (1, 64), (64, 65), (65, T)]:
    cache.append(k_all[:, :, lo:hi], v_all[:, :, lo:hi])
print("cache.len =", cache.len, "(expect", T, ")")

q = (torch.randn(H, D) * 0.5).to(torch.float16).to(DEV)

# ---- paged path ----
out_paged = cache.attention(q, softmax_scale=1.0 / (D ** 0.5))       # [H, D]

# ---- dense reference on the same quantized values ----
# Dequantize the cache content by hand (mirrors the kernel's math).
def dequant_k():
    k = torch.zeros(1, H, T, D, device=DEV, dtype=torch.float32)
    for blk in range((T + 127) // 128):
        for w in range(16):                       # 128/8 words
            word = cache.k_q[blk, :, :, w].to(torch.int64) & 0xFFFFFFFF  # [H, D]
            for nib in range(8):
                slot = blk * 128 + w * 8 + nib
                if slot >= T:
                    continue
                val = (word >> (4 * nib)) & 0xF                       # [H, D]
                val = torch.where(val >= 8, val - 16, val).float()
                gi = (w * 8 + nib) // G
                k[0, :, slot, :] = val * cache.k_s[blk, :, :, gi].float()
    return k

def dequant_v():
    v = torch.zeros(1, H, T, D, device=DEV, dtype=torch.float32)
    for blk in range((T + 127) // 128):
        for t in range(128):
            slot = blk * 128 + t
            if slot >= T:
                continue
            word = cache.v_q[blk, :, t, :].to(torch.int64) & 0xFFFFFFFF   # [H, D/8]
            for nib in range(8):
                val = (word >> (4 * nib)) & 0xF
                val = torch.where(val >= 8, val - 16, val).float()        # [H, D/8]
                d_idx = torch.arange(16, device=DEV) * 8 + nib            # channels
                gi = d_idx // G
                sc = cache.v_s[blk, :, t, :].float()[:, gi]               # [H, D/8]
                v[0, :, slot, d_idx] = val * sc
    return v

k_dq, v_dq = dequant_k(), dequant_v()
scores = torch.einsum("hd,htd->ht", q.float(), k_dq[0]) / (D ** 0.5)
probs = torch.softmax(scores, dim=-1)
ref = torch.einsum("ht,htd->hd", probs, v_dq[0])

diff = (out_paged.float() - ref).abs()
print(f"paged vs dense:  max {diff.max().item():.4f}  mean {diff.mean().item():.5f}")
print(f"ref magnitude:   mean {ref.abs().mean().item():.4f}")
rel = diff.mean() / ref.abs().mean()
print(f"relative:        {rel.item():.4%}")
print("PASS" if diff.max().item() < 0.15 else "FAIL")
