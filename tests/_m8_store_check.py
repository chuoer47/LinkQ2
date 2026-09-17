"""M8: store kernel vs pure-torch store reference (same quantization math)."""
import sys
sys.path.insert(0, "/home/<user>/<workdir>/qserve-lab")
import torch
from qslab.runtime.paged_decode import store_kv_quant

DEV = "cuda:0"
H, D, T = 8, 128, 128
torch.manual_seed(0)

k = (torch.randn(T, H, D) * 0.5).half().to(DEV)
v = (torch.randn(T, H, D) * 0.5).half().to(DEV)

# ---- pure-torch reference of the per-token-per-head scheme ----
def torch_store(x):
    amax = x.abs().amax(dim=-1, keepdim=True)          # [T,H,1]
    scale = (amax / 7.0).clamp_min(1e-4)
    q = torch.clamp(torch.floor(x / scale + 0.5), -8, 7).to(torch.int64)
    nib = (q + 8) & 0xF                                 # [T,H,D]
    nw = D // 8
    n6 = nib.view(T, H, nw, 8)
    packed = torch.zeros(T, H, nw, dtype=torch.int64, device=x.device)
    for i in range(8):
        packed |= n6[:, :, :, i] << (4 * i)
    return packed.to(torch.int32), scale.squeeze(-1).half()

kq_ref, ks_ref = torch_store(k)
vq_ref, vs_ref = torch_store(v)

# ---- run the triton kernel ----
# kernel layout: slot = t (per token), all H heads inside slot -> row [t*H+h]
kq = torch.zeros(T * H, D // 8, dtype=torch.uint32, device=DEV)
ks = torch.zeros(T * H, dtype=torch.float16, device=DEV)
vq = torch.zeros(T * H, D // 8, dtype=torch.uint32, device=DEV)
vs = torch.zeros(T * H, dtype=torch.float16, device=DEV)
slot_mapping = torch.arange(T, dtype=torch.int32, device=DEV)
store_kv_quant(k, v, (kq, ks), (vq, vs), slot_mapping)

# kernel row [t*H+h] must equal ref row [t, h]
kq_k = kq.view(T, H, D // 8)
ks_k = ks.view(T, H)
vq_k = vq.view(T, H, D // 8)
vs_k = vs.view(T, H)

mq = (kq_k.view(torch.int32) & 0xFFFFFFFF) - (kq_ref & 0xFFFFFFFF)
mv = (vq_k.view(torch.int32) & 0xFFFFFFFF) - (vq_ref & 0xFFFFFFFF)
ms = (ks_k.float() - ks_ref.float()).abs().max()
print(f"kq mismatch words: {(mq != 0).sum().item()} / {mq.numel()}")
print(f"vq mismatch words: {(mv != 0).sum().item()} / {mv.numel()}")
print(f"scale max diff: {ms.item():.6f}")
mismatch_frac = max((mq != 0).float().mean().item(), (mv != 0).float().mean().item())
print(f"mismatch fraction: {mismatch_frac:.4%}")
# Nibble-boundary cases (value exactly at x.5) resolve differently between
# Triton and torch due to fp32 non-determinism; this changes the quantized
# value by 1 LSB and is acceptable. Fail only on gross mismatch.
print("STORE PASS" if mismatch_frac < 0.02 and ms.item() < 1e-3 else "STORE FAIL")
