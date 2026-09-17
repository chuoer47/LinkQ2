"""Dump the first mismatching word's channels."""
import sys
sys.path.insert(0, "/home/<user>/<workdir>/qserve-lab")
import torch
from qslab.runtime.paged_decode import store_kv_quant

DEV = "cuda:0"
H, D, T = 8, 128, 128
torch.manual_seed(0)
k = (torch.randn(T, H, D) * 0.5).half().to(DEV)

kq = torch.zeros(T * H, D // 8, dtype=torch.uint32, device=DEV)
ks = torch.zeros(T * H, dtype=torch.float16, device=DEV)
slot_mapping = torch.arange(T, dtype=torch.int32, device=DEV)
store_kv_quant(k, v_dummy if (v_dummy := torch.randn(T, H, D).half().to(DEV)) is not None else v_dummy,
               (kq, ks), (torch.zeros_like(kq), torch.zeros_like(ks)), slot_mapping)

# torch ref for K
amax = k.abs().amax(dim=-1, keepdim=True)
scale = (amax / 7.0).clamp_min(1e-4)
qi = torch.clamp(torch.floor(k.float() / scale + 0.5), -8, 7).to(torch.int64)
nib = (qi + 8) & 0xF
n6 = nib.view(T, H, 16, 8)
ref = torch.zeros(T, H, 16, dtype=torch.int64, device=DEV)
for i in range(8):
    ref |= n6[:, :, :, i] << (4 * i)

kq_k = kq.view(T, H, 16).to(torch.int64) & 0xFFFFFFFF
mism = (kq_k != ref)
idx = mism.nonzero()[0]
t, h, w = idx.tolist()
print(f"first mismatch at t={t} h={h} w={w}")
print("kernel:", kq_k[t, h, w].item() & 0xFFFFFFFF, "| ref:", ref[t, h, w].item())
for nib in range(8):
    kq_val = (kq_k[t, h, w].item() >> (4 * nib)) & 0xF
    ref_val = (ref[t, h, w].item() >> (4 * nib)) & 0xF
    qi_ref = int(qi[t, h, w * 8 + nib].item())
    print(f"  ch{w*8+nib}: kernel_nib={kq_val} ref_nib={ref_val} qi={qi_ref} nib_ref={(qi_ref+8)&0xF}")
