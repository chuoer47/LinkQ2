"""Quantized paged KV — layout, encoding and decode correctness."""
import pytest
import torch

from qslab.runtime.model.paged_decode import store_kv_quant, paged_attention_decode

pytestmark = pytest.mark.gpu

DEV = "cuda"
H_Q, H_KV, D, BN = 16, 8, 128, 128
MAX_SLOTS = 8 * BN
V_GROUP = 64


def _pool(static_k=True):
    kq = torch.zeros(MAX_SLOTS, H_KV, D // 8, dtype=torch.uint32, device=DEV)
    if static_k:
        ks = (torch.rand(H_KV, D, device=DEV) * 0.3 + 0.1).half()   # [H, D]
    else:
        ks = torch.zeros(MAX_SLOTS, H_KV, dtype=torch.float16, device=DEV)
    vq = torch.zeros(MAX_SLOTS, H_KV, D // 8, dtype=torch.uint32, device=DEV)
    vs = torch.zeros(MAX_SLOTS, H_KV, D // V_GROUP, dtype=torch.float16, device=DEV)
    return (kq, ks), (vq, vs)


def _unpack(packed):
    """[S, H, NW] int64 words -> [S, H, D] offset-binary values."""
    bits = torch.stack([(packed >> (4 * g)) & 0xF for g in range(8)], -1)
    return bits.reshape(packed.shape[0], packed.shape[1], -1) - 8


def test_store_encoding_matches_reference():
    """Regression: store writes offset-binary (qi+8) and readers must decode the same way."""
    torch.manual_seed(0)
    T = 12
    k = (torch.randn(T, H_KV, D) * 2).half().to(DEV)
    v = (torch.randn(T, H_KV, D) * 2).half().to(DEV)

    kq = torch.zeros(MAX_SLOTS, H_KV, D // 8, dtype=torch.uint32, device=DEV)
    ks = (torch.rand(H_KV, D, device=DEV) * 0.3 + 0.1).half()          # static
    vq = torch.zeros(MAX_SLOTS, H_KV, D // 8, dtype=torch.uint32, device=DEV)
    vs = torch.zeros(MAX_SLOTS, H_KV, D // V_GROUP, dtype=torch.float16, device=DEV)

    slots = torch.arange(BN, BN + T, dtype=torch.int32, device=DEV)
    store_kv_quant(k, v, (kq, ks), (vq, vs), slots, v_group=V_GROUP)

    sl = slots.long()
    k_vals = _unpack((kq.index_select(0, sl).view(torch.int32).long() & 0xFFFFFFFF)
                     .view(T, H_KV, D // 8)).float()
    k_dq = k_vals * ks.float()[None, :, :]
    k_ref = torch.clamp(torch.floor(k.float() / ks.float()[None] + 0.5), -8, 7) * ks.float()[None]
    over = ((k_dq - k_ref).abs() > 0.55 * ks.float()[None] + 1e-3)
    assert over.float().mean().item() < 1e-3, f"{over.sum().item()} K positions off by >1 quantum"

    v_vals = _unpack((vq.index_select(0, sl).view(torch.int32).long() & 0xFFFFFFFF)
                     .view(T, H_KV, D // 8)).float().view(T, H_KV, D // V_GROUP, V_GROUP)
    v_dq = (v_vals * vs.index_select(0, sl).float()[:, :, :, None]).reshape(T, H_KV, D)
    vg = v.float().view(T, H_KV, D // V_GROUP, V_GROUP)
    vs_ref = (vg.abs().amax(-1, keepdim=True) / 7).clamp_min(1e-4)
    v_ref = (torch.clamp(torch.floor(vg / vs_ref + 0.5), -8, 7) * vs_ref).reshape(T, H_KV, D)
    vb = vs_ref.expand(T, H_KV, D // V_GROUP, V_GROUP).reshape(T, H_KV, D)
    over = ((v_dq - v_ref).abs() > 0.55 * vb + 1e-3)
    assert over.float().mean().item() < 1e-3, f"{over.sum().item()} V positions off by >1 quantum"


def test_decode_batched_block_tables():
    """Batched grid: each sequence reads its own block table and length."""
    torch.manual_seed(0)
    T1, T2 = 200, 300
    total = T1 + T2
    blocks = torch.tensor([[1, 2, 2], [3, 4, 5]], dtype=torch.int32, device=DEV)
    k = (torch.randn(total, H_KV, D) * 2).half().to(DEV)
    v = (torch.randn(total, H_KV, D) * 2).half().to(DEV)
    kq = torch.zeros(MAX_SLOTS, H_KV, D // 8, dtype=torch.uint32, device=DEV)
    ks = (torch.rand(H_KV, D, device=DEV) * 0.3 + 0.1).half()
    vq = torch.zeros(MAX_SLOTS, H_KV, D // 8, dtype=torch.uint32, device=DEV)
    vs = torch.zeros(MAX_SLOTS, H_KV, D // V_GROUP, dtype=torch.float16, device=DEV)

    slots = []
    for blks, n in ((blocks[0], T1), (blocks[1], T2)):
        t = torch.arange(n, device=DEV)
        slots.append(blks[t // BN] * BN + t % BN)
    slots = torch.cat(slots).to(torch.int32)
    store_kv_quant(k, v, (kq, ks), (vq, vs), slots, v_group=V_GROUP)

    lens = torch.tensor([T1, T2], dtype=torch.int32, device=DEV)
    q = (torch.randn(2, H_Q, D) * 2).half().to(DEV)
    out = paged_attention_decode(q, (kq, ks), (vq, vs), blocks, lens,
                                 block_n=BN, num_kv_heads=H_KV, v_group=V_GROUP)

    sl = slots.long()
    k_vals = _unpack((kq.index_select(0, sl).view(torch.int32).long() & 0xFFFFFFFF)
                     .view(total, H_KV, D // 8)).float()
    k_dq = (k_vals * ks.float()[None, :, :])
    v_vals = _unpack((vq.index_select(0, sl).view(torch.int32).long() & 0xFFFFFFFF)
                     .view(total, H_KV, D // 8)).float().view(total, H_KV, D // V_GROUP, V_GROUP)
    v_dq = (v_vals * vs.index_select(0, sl).float()[:, :, :, None]).reshape(total, H_KV, D)

    rep = H_Q // H_KV
    off = 0
    for b, n in enumerate((T1, T2)):
        kb = k_dq[off:off + n].permute(1, 0, 2).repeat_interleave(rep, 0)
        vb = v_dq[off:off + n].permute(1, 0, 2).repeat_interleave(rep, 0)
        sc = torch.einsum("hd,htd->ht", q[b].float(), kb) / (D ** 0.5)
        ref = torch.einsum("ht,htd->hd", torch.softmax(sc, -1), vb)
        gap = (out[b].float() - ref).abs().max().item()
        assert gap < 2.0, f"seq {b} routed to the wrong blocks (gap {gap:.3f})"
        off += n

    # and the shorter sequence must not see the longer one's tail
    swapped = lens.flip(0)
    out2 = paged_attention_decode(q, (kq, ks), (vq, vs), blocks, swapped,
                                  block_n=BN, num_kv_heads=H_KV, v_group=V_GROUP)
    assert (out2[0].float() - out[0].float()).abs().max().item() > 1e-3, \
        "sequence 0 unchanged when its context length was swapped"
