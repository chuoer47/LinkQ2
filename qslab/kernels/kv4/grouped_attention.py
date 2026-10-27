"""Grouped-query KV4 verification with FP16 tensor-core inputs and FP32 accumulation."""
import triton
import triton.language as tl


@triton.jit
def kv4_grouped_verify_kernel(qp, kp, ks, vp, vs, bt, lengths, out,
        HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, M: tl.constexpr,
        PAGE: tl.constexpr, PAGES: tl.constexpr, GROUP: tl.constexpr,
        STATIC_K: tl.constexpr, SPLITS: tl.constexpr, Q_STRIDE: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr):
    head = tl.program_id(0)
    seq = tl.program_id(1)
    split = tl.program_id(2) % SPLITS
    query_block = tl.program_id(2) // SPLITS
    rep: tl.constexpr = HQ // HK
    flat_q = query_block * BM + tl.arange(0, BM)
    m = flat_q // rep
    qhead = head * rep + flat_q % rep
    row = seq * M + m
    d = tl.arange(0, D)
    ns = tl.arange(0, BN)
    q = tl.load(qp + row[:, None] * Q_STRIDE + qhead[:, None] * D + d[None, :],
                (m < M)[:, None], other=0)
    length = tl.load(lengths + seq)
    tiles = tl.cdiv(tl.minimum(length + M - 1, PAGES * PAGE), BN)
    per_split = tl.cdiv(tiles, SPLITS)
    maximum = tl.full((BM,), -float('inf'), tl.float32)
    denom = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    if STATIC_K:
        key_scale = tl.load(ks + head * D + d).to(tl.float32)
    for tile in range(split * per_split, tl.minimum((split + 1) * per_split, tiles)):
        pos = tile * BN + ns
        page = tl.load(bt + seq * PAGES + pos // PAGE)
        slot = tl.maximum(page, 0) * PAGE + pos % PAGE
        word = (slot[:, None] * HK + head) * (D // 8) + d[None, :] // 8
        key = ((tl.load(kp + word) >> ((d[None, :] % 8) * 4)) & 15).to(tl.int32) - 8
        if STATIC_K:
            key = key.to(tl.float32) * key_scale[None, :]
        else:
            key = key.to(tl.float32) * tl.load(ks + slot * HK + head).to(tl.float32)[:, None]
        scores = tl.dot(q, tl.trans(key.to(tl.float16))) * (D ** -0.5)
        valid = (pos[None, :] < length + m[:, None]) & (page[None, :] >= 0) & (m < M)[:, None]
        scores = tl.where(valid, scores, -float('inf'))
        next_max = tl.maximum(maximum, tl.max(scores, 1))
        safe_max = tl.where(next_max == -float('inf'), 0., next_max)
        prob = tl.exp(scores - safe_max[:, None])
        alpha = tl.exp(maximum - safe_max)
        value = ((tl.load(vp + word) >> ((d[None, :] % 8) * 4)) & 15).to(tl.int32) - 8
        value_scale = tl.load(vs + (slot[:, None] * HK + head) * (D // GROUP) + d[None, :] // GROUP).to(tl.float32)
        value = value.to(tl.float32) * value_scale
        acc = acc * alpha[:, None] + tl.dot(prob.to(tl.float16), value.to(tl.float16))
        denom = denom * alpha + tl.sum(prob, 1)
        maximum = next_max
    if SPLITS == 1:
        tl.store(out + (row[:, None] * HQ + qhead[:, None]) * D + d[None, :],
                  acc / tl.maximum(denom[:, None], 1e-20), (m < M)[:, None])
    else:
        base = ((row * HQ + qhead) * SPLITS + split) * (D + 2)
        tl.store(out + base[:, None] + d[None, :], acc, (m < M)[:, None])
        tl.store(out + base + D, maximum, m < M)
        tl.store(out + base + D + 1, denom, m < M)
