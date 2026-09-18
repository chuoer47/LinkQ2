"""M9: the M-query verify kernel against a torch reference.

The verify forward sends M = gamma+1 queries per sequence through the decode
kernel — row m reads keys [0, L+m), its own slot being the causal bound. The
reference dequantizes the same int4 pool and does plain softmax attention, so
any difference is kernel arithmetic, not quantization.

Causality is asserted structurally: corrupting the KV at the LAST draft slot
must leave rows 0..M-2 untouched (they read strictly below it), and the -1
block padding (a capped draft window) must degrade to attention over the
available prefix instead of reading wild memory.
"""
import pytest
import torch

pytestmark = pytest.mark.gpu

from qslab.runtime.paged_decode import (paged_attention_decode,
                                        store_kv_quant)

BLOCK = 32          # small blocks so multi-tile paths are exercised cheaply
V_GROUP = 64


def _make_pool(n_slots, hkv, d, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    k = (torch.randn(n_slots, hkv, d, generator=g, device="cuda") * 0.5).half()
    v = (torch.randn(n_slots, hkv, d, generator=g, device="cuda") * 0.5).half()
    # calibration-like per-channel scale (amax/7): no channel saturates the
    # int4 range, so the roundtrip error is bounded by half a quant step
    ks = (k.float().abs().amax(dim=0) / 7).clamp_min(1e-4).half()
    kq = torch.zeros(n_slots, hkv, d // 8, dtype=torch.uint32, device="cuda")
    vq = torch.zeros(n_slots, hkv, d // 8, dtype=torch.uint32, device="cuda")
    vs = torch.zeros(n_slots, hkv, d // V_GROUP, dtype=torch.float16, device="cuda")
    # store through the real quantizer so the pool holds genuine packed data
    slots = torch.arange(n_slots, dtype=torch.int32, device="cuda")
    store_kv_quant(k, v, (kq, ks), (vq, vs), slots, v_group=V_GROUP)
    return (kq, ks), (vq, vs)


def _unpack(words: torch.Tensor) -> torch.Tensor:
    """[..., NW] uint32 -> [..., NW*8] signed nibbles."""
    shifts = torch.arange(8, device=words.device) * 4
    nib = ((words.unsqueeze(-1).int() >> shifts) & 0xF) - 8
    return nib.flatten(-2)


def _dequant(k_cache, v_cache, hkv, d):
    kq, ks = k_cache
    vq, vs = v_cache
    k = (_unpack(kq).float().reshape(-1, hkv, d)
         * ks.float()[None]).reshape(kq.shape[0], hkv, d)
    ng = d // V_GROUP
    v = _unpack(vq).float().reshape(-1, hkv, ng, V_GROUP) * vs.float()[:, :, None]
    return k, v.reshape(vq.shape[0], hkv, d)


def _reference(k, v, q, block_tables, context_lens, M, bs, hq, hkv, d, scale):
    """q [bs, M, hq, d]; returns [bs, M, hq, d]."""
    rep = hq // hkv
    out = torch.empty_like(q)
    for i in range(bs):
        L = int(context_lens[i])
        for m in range(M):
            # positions 0..L+m-1, but never past this seq's real blocks
            n_real = int((block_tables[i] >= 0).sum()) * BLOCK
            hi = min(L + m, n_real)
            slots = []
            for p in range(hi):
                slots.append(int(block_tables[i, p // BLOCK]) * BLOCK + p % BLOCK)
            K = k[slots]                                     # [T, hkv, d]
            V = v[slots]
            Qi = q[i, m]                                     # [hq, d]
            # GQA: q head h attends kv head h // rep; expand K/V to q heads
            heads = torch.arange(hq) // rep
            Kh = K.float()[:, heads, :]                      # [T, hq, d]
            Vh = V.float()[:, heads, :]                      # [T, hq, d]
            scores = torch.einsum("hd,thd->ht", Qi.float(), Kh) * scale
            p = torch.softmax(scores, dim=-1)                # [hq, T]
            o = torch.einsum("ht,thd->hd", p.float(), Vh)
            out[i, m] = o.half()
    return out


def _run_case(bs, L, M, hq, hkv, d, seed=0):
    n_tokens = bs * (L + M + 8)
    n_slots = n_tokens + BLOCK * 4
    torch.manual_seed(seed)
    # each seq gets its own disjoint random blocks
    block_tables = []
    free = list(range(0, n_slots // BLOCK))
    for _ in range(bs):
        n_blocks = (L + M + BLOCK - 1) // BLOCK + 1
        ids = free[:n_blocks]
        free = free[n_blocks:]
        block_tables.append(ids + [-1] * (20 - len(ids)))
    bt = torch.tensor(block_tables, dtype=torch.int32, device="cuda")
    ctx = torch.tensor([L] * bs, dtype=torch.int32, device="cuda")

    k_cache, v_cache = _make_pool(n_slots, hkv, d, seed)
    k, v = _dequant(k_cache, v_cache, hkv, d)

    g = torch.Generator(device="cuda").manual_seed(seed + 1)
    q = (torch.randn(bs, M, hq, d, generator=g, device="cuda") * 0.5).half()
    # fill the slots the sequences will actually read (incl. draft slots)
    for i in range(bs):
        for p in range(L + M):
            slot = int(bt[i, p // BLOCK]) * BLOCK + p % BLOCK
            store_kv_quant(k[slot][None].expand(1, hkv, d).contiguous(),
                           v[slot][None].expand(1, hkv, d).contiguous(),
                           k_cache, v_cache,
                           torch.tensor([slot], dtype=torch.int32, device="cuda"),
                           v_group=V_GROUP)
    k, v = _dequant(k_cache, v_cache, hkv, d)   # re-dequant with the new stores

    out = paged_attention_decode(
        q.reshape(bs * M, hq, d).contiguous(), k_cache, v_cache, bt, ctx,
        block_n=BLOCK, num_kv_heads=hkv, v_group=V_GROUP, verify_m=M)
    out = out.reshape(bs, M, hq, d).float()
    ref = _reference(k, v, q, bt, ctx, M, bs, hq, hkv, d, d ** -0.5).float()
    return out, ref


@pytest.mark.parametrize("bs,L,M", [(1, 5, 4), (2, 40, 4), (3, 130, 5),
                                    (2, 300, 2), (1, 5, 1)])
def test_verify_matches_reference(bs, L, M):
    out, ref = _run_case(bs, L, M, hq=4, hkv=2, d=64)
    gap = (out - ref).abs()
    assert gap.max().item() < 2e-3, f"max|Δ| = {gap.max().item():.5f}"


def test_causality_last_draft_slot():
    """Rows below the last draft slot must not see its KV at all."""
    bs, L, M, hq, hkv, d = 1, 40, 4, 4, 2, 64
    # corrupt the KV at the last draft position (row M-1's own slot)
    k_cache, v_cache = _make_pool(64, hkv, d, seed=3)
    # positions map 1:1 to slots: position p lives in block p // BLOCK
    bt = torch.arange(20, dtype=torch.int32, device="cuda")[None]
    ctx = torch.tensor([L], dtype=torch.int32, device="cuda")
    g = torch.Generator(device="cuda").manual_seed(9)
    q = (torch.randn(bs, M, hq, d, generator=g, device="cuda") * 0.5).half()

    def run():
        return paged_attention_decode(
            q.reshape(bs * M, hq, d).contiguous(), k_cache, v_cache, bt, ctx,
            block_n=BLOCK, num_kv_heads=hkv, v_group=V_GROUP, verify_m=M
        ).reshape(bs, M, hq, d)

    a = run()
    # overwrite the slot of the LAST draft (position L+M-2, row M-1's own
    # slot; the bonus position L+M-1 is never read by any row)
    bad_k = torch.randn(1, hkv, d, device="cuda").half() * 10
    bad_v = torch.randn(1, hkv, d, device="cuda").half() * 10
    slot = L + M - 2
    store_kv_quant(bad_k, bad_v, k_cache, v_cache,
                   torch.tensor([slot], dtype=torch.int32, device="cuda"),
                   v_group=V_GROUP)
    b = run()
    assert torch.equal(a[0, :M - 1], b[0, :M - 1]), "earlier rows changed"
    assert not torch.equal(a[0, M - 1], b[0, M - 1]), "last row did not move"


def test_neg1_padding_degrades_to_prefix():
    """A row past its sequence's blocks reads -1 padding: finite output,
    equal to attention over the available prefix."""
    bs, L, M, hq, hkv, d = 1, 100, 4, 4, 2, 64
    n_slots = 96                     # blocks 0..2 must exist in the pool
    k_cache, v_cache = _make_pool(n_slots, hkv, d, seed=5)
    k, v = _dequant(k_cache, v_cache, hkv, d)
    # only 3 real blocks = 96 slots covered; row 0 already needs 4 tiles
    bt = torch.tensor([[0, 1, 2] + [-1] * 17], dtype=torch.int32, device="cuda")
    ctx = torch.tensor([L], dtype=torch.int32, device="cuda")
    g = torch.Generator(device="cuda").manual_seed(6)
    q = (torch.randn(bs, M, hq, d, generator=g, device="cuda") * 0.5).half()

    out = paged_attention_decode(
        q.reshape(bs * M, hq, d).contiguous(), k_cache, v_cache, bt, ctx,
        block_n=BLOCK, num_kv_heads=hkv, v_group=V_GROUP, verify_m=M)
    out = out.reshape(bs, M, hq, d).float()
    assert torch.isfinite(out).all()

    # reference: attention over the first 96 slots for every row
    ref = torch.empty_like(q)
    for m in range(M):
        K, V = k[:96], v[:96]
        heads = torch.arange(hq) // (hq // hkv)
        Kh = K.float()[:, heads, :]                            # [T, hq, d]
        Vh = V.float()[:, heads, :]                            # [T, hq, d]
        scores = torch.einsum("hd,thd->ht", q[0, m].float(), Kh) * d ** -0.5
        p = torch.softmax(scores, dim=-1)
        o = torch.einsum("ht,thd->hd", p.float(), Vh)
        ref[0, m] = o.half()
    gap = (out - ref.float()).abs()
    assert gap.max().item() < 2e-3, f"max|Δ| = {gap.max().item():.5f}"


def test_materialize_kv_roundtrip():
    """store -> materialize must return the original values within the int4
    quantization step (the prefix-cache hit path depends on it)."""
    hkv, d, n = 2, 64, 96
    g = torch.Generator(device="cuda").manual_seed(11)
    k = (torch.randn(n, hkv, d, generator=g, device="cuda") * 0.5).half()
    v = (torch.randn(n, hkv, d, generator=g, device="cuda") * 0.5).half()
    # calibration-like per-channel scale (amax/7): no channel saturates the
    # int4 range, so the roundtrip error is bounded by half a quant step
    ks = (k.float().abs().amax(dim=0) / 7).clamp_min(1e-4).half()
    kq = torch.zeros(n, hkv, d // 8, dtype=torch.uint32, device="cuda")
    vq = torch.zeros(n, hkv, d // 8, dtype=torch.uint32, device="cuda")
    vs = torch.zeros(n, hkv, d // V_GROUP, dtype=torch.float16, device="cuda")
    slots = torch.arange(n, dtype=torch.int64, device="cuda")
    store_kv_quant(k, v, (kq, ks), (vq, vs), slots, v_group=V_GROUP)
    from qslab.runtime.paged_decode import materialize_kv
    km, vm = materialize_kv((kq, ks), (vq, vs), slots, v_group=V_GROUP)
    # error bounded by the quantization step: half a step for round-to-nearest
    k_step = ks.float().repeat(n, 1, 1)
    assert (km.float() - k.float()).abs().max() <= 0.51 * k_step.max()
    assert (vm.float() - v.float()).abs().max() <= 0.51 * (v.float().abs() / 7).max() * 1.01
    assert (km - k).abs().float().mean() < 0.05
