"""Direct store->decode consistency test, bypassing the engine."""
import sys
sys.path.insert(0, "/home/<user>/<workdir>/qserve-lab")
import torch
from qslab.runtime.paged_decode import store_kv_quant, paged_attention_decode

DEV = "cuda:0"
H_Q, H_KV, D, BN = 16, 8, 128, 128
T = 300
torch.manual_seed(0)

k = (torch.randn(T, H_KV, D) * 0.5).half().to(DEV)
v = (torch.randn(T, H_KV, D) * 0.5).half().to(DEV)

total_slots = 512
kq = torch.zeros(total_slots, H_KV, D // 8, dtype=torch.uint32, device=DEV)
ks = torch.zeros(total_slots, H_KV, dtype=torch.float16, device=DEV)
vq = torch.zeros(total_slots, H_KV, D // 8, dtype=torch.uint32, device=DEV)
vs = torch.zeros(total_slots, H_KV, dtype=torch.float16, device=DEV)

# block table: identity (block b -> slot base b*BN)
bt = torch.arange(T // BN + 1, dtype=torch.int32, device=DEV)  # block ids
# slot for token t = t (block b = t//BN, offset t%BN -> slot base bt[b]+off = t)
slot_mapping = torch.arange(T, dtype=torch.int32, device=DEV)

store_kv_quant(k, v, (kq, ks), (vq, vs), slot_mapping)

q = (torch.randn(H_Q, D) * 0.5).half().to(DEV)
ctx = T
out = paged_attention_decode(q, (kq, ks), (vq, vs), bt, ctx, block_n=BN,
                             num_kv_heads=H_KV)

# dense reference: dequant all, GQA matmul
def dequant(packed, scales):
    rows = packed.shape[0]
    q = packed.view(rows, H_KV, D // 8).to(torch.int64) & 0xFFFFFFFF
    shifts = (torch.arange(8, device=DEV) * 4)[None, None, None, :]
    nibs = ((q[:, :, :, None] >> shifts) & 0xF).to(torch.int32)
    nibs = torch.where(nibs >= 8, nibs - 16, nibs).float()
    sc = scales.view(rows, H_KV).float()[:, :, None, None]
    return (nibs * sc).reshape(rows, H_KV, D)

k_dq = dequant(kq, ks)          # [slots, H_KV, D]
v_dq = dequant(vq, vs)
# slots used: slot t (per token), so gather k_dq[t] for t in range(T)
k_ref = k_dq[:T].permute(1, 0, 2)                 # [H_KV, T, D]
v_ref = v_dq[:T].permute(1, 0, 2)
n_rep = H_Q // H_KV
k_rep = k_ref.repeat_interleave(n_rep, dim=0)     # [H_Q, T, D]
v_rep = v_ref.repeat_interleave(n_rep, dim=0)
scores = torch.einsum("hd,htd->ht", q.float(), k_rep) / (D ** 0.5)
probs = torch.softmax(scores, dim=-1)
ref = torch.einsum("ht,htd->hd", probs, v_rep)

d = (out.float() - ref).abs()
print(f"max diff {d.max().item():.5f} | mean {d.mean().item():.6f} | ref mean {ref.abs().mean().item():.4f}")
print("STORE-DECODE CONSISTENT" if d.max().item() < 0.1 else "INCONSISTENT")
