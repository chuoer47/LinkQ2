"""L1: KV4 paged cache — int4 KV stored in fixed-size blocks.

Block-based storage so attention can stream the context block by block
(dequantizing each tile in registers) instead of materializing a dense FP16
copy of the whole cache. See docs/design-m7.md.

Layout per block (BLOCK_N tokens), for one layer:
  k_q [nb, H, D, BLOCK_N/8]     uint32   K per-channel, token-major packs
  k_s [nb, H, D, BLOCK_N/GROUP] fp16
  v_q [nb, H, BLOCK_N, D/8]     uint32   V per-token, channel-major packs
  v_s [nb, H, BLOCK_N, D/GROUP] fp16

BLOCK_N is a multiple of GROUP so a quantization group never spans blocks.

The cache owns a logical->physical block table. Blocks are handed out in
order (append-only decode); a caller that needs eviction would add a policy
here without touching the kernel.
"""
from __future__ import annotations

import torch

PACK_G = 8
_MIN_SCALE = 1e-4   # above fp16 smallest normal (6.1e-5): avoids underflow-to-zero


class KV4PagedCache:
    """One layer's paged int4 KV storage."""

    def __init__(self, batch: int, num_kv_heads: int, head_dim: int,
                 max_len: int, device: str, group: int = 64,
                 block_n: int = 128):
        assert block_n % group == 0, "block must contain whole quant groups"
        assert head_dim % PACK_G == 0 and head_dim % group == 0
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.group = group
        self.block_n = block_n
        self.device = device
        self.len = 0                                    # valid tokens across all blocks

        self.num_blocks = (max_len + block_n - 1) // block_n
        nb, H, D = self.num_blocks, num_kv_heads, head_dim
        nk_w, nk_g = block_n // PACK_G, block_n // group
        nv_w, nv_g = D // PACK_G, D // group

        self.k_q = torch.zeros(nb, H, D, nk_w, device=device, dtype=torch.uint32)
        self.k_s = torch.zeros(nb, H, D, nk_g, device=device, dtype=torch.float16)
        self.v_q = torch.zeros(nb, H, block_n, nv_w, device=device, dtype=torch.uint32)
        self.v_s = torch.zeros(nb, H, block_n, nv_g, device=device, dtype=torch.float16)

        # logical -> physical block map. Identity for append-only decode.
        self.block_table = torch.arange(self.num_blocks, device=device,
                                        dtype=torch.int32)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def append(self, k_new: torch.Tensor, v_new: torch.Tensor):
        """Append [1, H, T, D] fp16 tokens. T may cross a block boundary."""
        T = k_new.shape[2]
        start = self.len
        pos = 0
        while pos < T:
            blk = (start + pos) // self.block_n
            off = (start + pos) % self.block_n
            take = min(self.block_n - off, T - pos)
            self._write_slice(blk, off, k_new[:, :, pos:pos + take],
                              v_new[:, :, pos:pos + take])
            pos += take
        self.len = start + T

    @torch.no_grad()
    def _write_slice(self, blk: int, off: int, k, v):
        """Write tokens [off, off+T) of one block; k,v are [1, H, T, D] fp16."""
        # ---- K: per-channel along the token axis ----
        kt = k[0].permute(0, 2, 1).contiguous()           # [H, T, D] -> [H, D, T]
        self._write_k(blk, off, kt)
        # ---- V: per-token along D ----
        vt = v[0].contiguous()                            # [H, T, D]
        self._write_v(blk, off, vt)

    @torch.no_grad()
    def _write_k(self, blk: int, off: int, kt: torch.Tensor):
        """kt [H, D, T] fp32/fp16 -> quantize per (channel, group) and pack."""
        g = self.group
        H, D, T = kt.shape
        x = kt.to(torch.float32)
        # group along the token axis; groups are aligned to `off` since block_n
        # is a multiple of g and `off` advances in g-aligned steps within a block
        # only when T is a multiple of g. Decode appends 1 token, so we
        # re-quantize the affected groups including previously written tokens.
        for gi_local in range((off % g + T + g - 1) // g):
            gi = (off // g) + gi_local
            tok_lo = gi * g                     # absolute token index in block
            tok_hi = min(tok_lo + g, self.block_n)
            have_lo = off
            have_hi = off + T
            seg_lo = max(tok_lo, have_lo)
            seg_hi = min(tok_hi, have_hi)
            if seg_hi <= seg_lo:
                continue
            # gather the segment's tokens for this group (may include earlier
            # tokens already in the block when a group is partially filled)
            seg = x[:, :, seg_lo - off:seg_hi - off]              # [H, D, seg]
            amax = seg.abs().amax(dim=-1, keepdim=True)           # [H, D, 1]
            scale = (amax / 7.0).clamp_min(_MIN_SCALE)
            q = torch.clamp(torch.round(seg / scale), -8, 7).to(torch.int64)
            nib = q & 0xF                                         # [H, D, seg]
            # write nibbles into the packed words (word = 8 tokens)
            for t in range(seg_hi - seg_lo):
                slot = seg_lo + t                     # absolute token index
                w, nib_i = slot // PACK_G, slot % PACK_G
                cur = self.k_q[blk, :, :, w].to(torch.int64) & 0xFFFFFFFF
                cleared = cur & ~(0xF << (4 * nib_i))
                self.k_q[blk, :, :, w] = (cleared | (nib[:, :, t] << (4 * nib_i))
                                          ).to(torch.int32).view(torch.uint32)
            self.k_s[blk, :, :, gi] = scale.squeeze(-1).to(torch.float16)

    @torch.no_grad()
    def _write_v(self, blk: int, off: int, vt: torch.Tensor):
        """vt [H, T, D] fp32/fp16 -> per-token group quantize along D."""
        g = self.group
        H, T, D = vt.shape
        x = vt.to(torch.float32)
        xg = x.reshape(H, T, D // g, g)
        amax = xg.abs().amax(dim=-1, keepdim=True)
        scale = (amax / 7.0).clamp_min(_MIN_SCALE)
        q = torch.clamp(torch.round(xg / scale), -8, 7).to(torch.int64)
        nib = (q & 0xF).reshape(H, T, D)                          # [H, T, D]
        # pack along D (8 per word)
        nw = D // PACK_G
        nw_nib = nib.reshape(H, T, nw, PACK_G)
        acc = torch.zeros(H, T, nw, dtype=torch.int64, device=nib.device)
        for i in range(PACK_G):
            acc |= nw_nib[:, :, :, i] << (4 * i)
        self.v_q[blk, :, off:off + T, :] = acc.to(torch.int32).view(torch.uint32)
        self.v_s[blk, :, off:off + T, :] = scale.squeeze(-1).to(torch.float16)

    # ------------------------------------------------------------------
    def memory_bytes_valid(self) -> int:
        nb_used = (self.len + self.block_n - 1) // self.block_n
        if nb_used == 0:
            return 0
        kq = self.k_q[:nb_used].nelement() * 4
        ks = self.k_s[:nb_used].nelement() * 2
        vq = self.v_q[:nb_used].nelement() * 4
        vs = self.v_s[:nb_used].nelement() * 2
        return kq + ks + vq + vs

    @property
    def memory_bytes(self) -> int:
        return (self.k_q.nelement() * 4 + self.k_s.nelement() * 2
                + self.v_q.nelement() * 4 + self.v_s.nelement() * 2)

    def reset(self):
        self.k_q.zero_()
        self.k_s.zero_()
        self.v_q.zero_()
        self.v_s.zero_()
        self.len = 0

    def attention(self, q_layer: torch.Tensor, softmax_scale: float):
        """q_layer [H, D] fp16 -> [H, D] fp16 via the paged int4 kernel."""
        from qslab.kernels.kv4_paged_attention import kv4_paged_attention
        return kv4_paged_attention(
            q_layer, (self.k_q, self.k_s), (self.v_q, self.v_s),
            self.block_table, self.len, block_n=self.block_n, group=self.group)
