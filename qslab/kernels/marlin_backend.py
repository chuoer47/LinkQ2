"""Marlin backend for W4Linear: v1 pack -> Marlin repack + gemm call."""
from __future__ import annotations

import torch

from qslab.kernels.marlin_ext import get_marlin

_SCALE_PERM = None
_PERM = None


def _get_perms():
    """Marlin's thread/tile permutation tables, verbatim from upstream; they must match the
    CUDA source exactly."""
    global _SCALE_PERM, _SCALE_PERM_SINGLE, _PERM
    if _PERM is not None:
        return _SCALE_PERM, _SCALE_PERM_SINGLE, _PERM
    import numpy as np

    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    _SCALE_PERM, _PERM = scale_perm, perm
    _SCALE_PERM_SINGLE = scale_perm_single
    return scale_perm, scale_perm_single, _PERM


def pack_v1_to_marlin(qfp: torch.Tensor, scale: torch.Tensor,
                      in_features: int, group_size: int = 128):
    """(v1 qfp [O, I/8] uint32, scale [O, I/g] fp16) -> (B, s, workspace)."""
    O, I = scale.shape[0], in_features
    assert group_size in (-1, 128), "marlin groupsize must be -1 or 128"
    assert I % 128 == 0 and O % 256 == 0, \
        f"marlin needs I%128==0, O%256==0, got {I}x{O}"
    scale_perm, scale_perm_single, perm = _get_perms()

    # 1) unpack nibbles -> signed q in [-8, 7], shape [O, I]
    qi = qfp.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    qn = torch.zeros(O, I, dtype=torch.uint8, device=qfp.device)
    for nib in range(8):
        qn[:, nib::8] = ((qi >> (4 * nib)) & 0xF).to(torch.uint8)
    q = qn.to(torch.int16)
    q = torch.where(q >= 8, q - 16, q)              # signed [-8, 7]

    # 2) marlin wants unsigned [0, 15] (its dequant adds -8 internally)
    w_unsigned = (q + 8).to(torch.uint8)            # [O, I]

    # 3) marlin pack() tile/perm sequence, on the [I, O] view (transposed)
    # (mirrors upstream Layer.pack: w [I,O] regrouped per-group then permuted)
    w = w_unsigned.t().contiguous()                  # [I, O]
    s = scale.t().contiguous()                       # [I/g, O]
    if group_size != I:
        w = w.reshape((-1, group_size, O))
        w = w.permute(1, 0, 2)
        w = w.reshape((group_size, -1))
        s = s.reshape((1, -1))
    w = w.reshape((group_size, -1, O))
    w = w.permute(1, 0, 2)
    w = w.reshape((I, O)).contiguous()
    s = s.reshape((-1, len(scale_perm)))[:, scale_perm] if group_size != I else \
        s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    s = s.reshape((-1, O)).contiguous()

    # 4) 16x16 tile permute
    tile = 16
    w = w.reshape((I // tile, tile, O // tile, tile))
    w = w.permute((0, 2, 1, 3))
    w = w.reshape((I // tile, O * tile))
    res = w.reshape((-1, perm.numel()))[:, perm].reshape(w.shape)

    # 5) pack 8 nibbles per int32 (marlin's order)
    res = res.to(torch.uint8) & 0xF
    q32 = torch.zeros(res.shape[0], res.shape[1] // 8, dtype=torch.int64,
                      device=res.device)
    for i in range(8):
        q32 |= res[:, i::8].to(torch.int64) << (4 * i)
    B = q32.to(torch.int32)
    return B, s.contiguous()


def marlin_gemm(x: torch.Tensor, B: torch.Tensor, s: torch.Tensor,
                workspace: torch.Tensor) -> torch.Tensor:
    mod = get_marlin()
    M, K = x.shape[0], x.shape[1]
    N = s.shape[1]
    C = torch.empty(M, N, dtype=x.dtype, device=x.device)
    # upstream sig: mul(A, B, C, s, workspace, thread_k=-1, thread_n=-1, sms=-1, max_par=16)
    mod.mul(x, B, C, s, workspace, -1, -1, -1, 16)
    return C
