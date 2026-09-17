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
# fp16 subnormals start around 6e-8; keeping the scale above that avoids a
# silent underflow-to-zero for all-zero groups (which would divide by zero
# in the softmax). 1e-4 is safely above fp16's smallest normal (6.1e-5).
_MIN_SCALE = 1e-4


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
            # NOTE: the whole group must be re-quantized against the tokens
            # already in the block. A per-group scale is shared, so writing a
            # new token with a scale derived from only the new tokens would
            # silently corrupt the earlier ones in the same group.
            #
            # Tokens already in the block are recovered from the packed data
            # (cheap: one group = at most `g` tokens across the whole block).
            new_lo = max(tok_lo, off) - off            # index into x
            new_hi = min(tok_hi, off + T) - off
            parts = []
            if tok_lo < off:                           # already-written prefix
                parts.append(self._dequant_k_group(blk, gi, tok_lo, off))
            if new_hi > new_lo:
                parts.append(x[:, :, new_lo:new_hi])
            if tok_hi > off + T:                       # already-written suffix
                parts.append(self._dequant_k_group(blk, gi, off + T, tok_hi))
            seg = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
            assert seg.shape[-1] == tok_hi - tok_lo, \
                f"group reconstruction size mismatch: {seg.shape[-1]} vs {tok_hi - tok_lo}"
            amax = seg.abs().amax(dim=-1, keepdim=True)           # [H, D, 1]
            scale = (amax / 7.0).clamp_min(_MIN_SCALE)
            q = torch.clamp(torch.round(seg / scale), -8, 7).to(torch.int64)
            nib = q & 0xF                                         # [H, D, seg]
            for t in range(tok_hi - tok_lo):
                slot = tok_lo + t                     # absolute token index
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
            self.block_table, self.len, block_n=self.block_n, group=self.group,
            num_kv_heads=self.num_kv_heads)

    # ------------------------------------------------------------------
    # Compatibility path: prefill still needs a dense FP16 view.
    #
    # The paged kernel is an M=1 (one query token per program) design; running
    # it for a T-token prefill would re-read the whole context T times. So
    # prefill materializes once (O(T), amortized over the whole request) and
    # decode goes through the paged kernel (no dense copy, int4 read width).
    @torch.no_grad()
    def update(self, k_new: torch.Tensor, v_new: torch.Tensor, start=None):
        """Append then return a dense fp16 view [1, H, len, D] (prefill path)."""
        self.append(k_new, v_new)
        return self.dense_view()

    @torch.no_grad()
    def _dequant_k_group(self, blk: int, gi: int, tok_lo: int, tok_hi: int):
        """Recover K values [H, D, tok_hi-tok_lo] for one group from the packs."""
        H, D, g = self.num_kv_heads, self.head_dim, self.group
        out = torch.zeros(H, D, tok_hi - tok_lo, device=self.device,
                          dtype=torch.float32)
        sc = self.k_s[blk, :, :, gi].float()                       # [H, D]
        for slot in range(tok_lo, tok_hi):
            w, nib_i = slot // PACK_G, slot % PACK_G
            word = self.k_q[blk, :, :, w].to(torch.int64) & 0xFFFFFFFF   # [H, D]
            val = (word >> (4 * nib_i)) & 0xF
            val = torch.where(val >= 8, val - 16, val).float()
            out[:, :, slot - tok_lo] = val * sc
        return out

    @torch.no_grad()
    def dense_view(self):
        """Dequantize the whole valid prefix back to fp16 (prefill only)."""
        n_used = self.len
        k = torch.zeros(1, self.num_kv_heads, n_used, self.head_dim,
                        device=self.device, dtype=torch.float16)
        v = torch.zeros_like(k)
        for b in range((n_used + self.block_n - 1) // self.block_n):
            lo = b * self.block_n
            hi = min(lo + self.block_n, n_used)
            k[:, :, lo:hi] = self._dequant_k_block(b)[:, :, :hi - lo]
            v[:, :, lo:hi] = self._dequant_v_block(b)[:, :, :hi - lo]
        return k, v

    def _dequant_k_block(self, blk: int):
        """[1, H, block_n, D] fp16 for one block."""
        H, D, BN, g = self.num_kv_heads, self.head_dim, self.block_n, self.group
        nw, ng = BN // PACK_G, BN // g
        q = self.k_q[blk].to(torch.int64) & 0xFFFFFFFF           # [H, D, nw]
        shifts = torch.arange(PACK_G, device=self.device) * 4
        val = (q.unsqueeze(-1) >> shifts) & 0xF                   # [H, D, nw, 8]
        val = val.to(torch.int32)
        val = torch.where(val >= 8, val - 16, val).float()
        # scale covers `g` consecutive tokens -> expand along the token axis
        s = self.k_s[blk].float().repeat_interleave(g, dim=-1)    # [H, D, BN]
        deq = val.reshape(H, D, BN) * s                           # [H, D, BN]
        return deq.permute(0, 2, 1).unsqueeze(0).to(torch.float16)   # [1,H,BN,D]

    def _dequant_v_block(self, blk: int):
        H, D, BN, g = self.num_kv_heads, self.head_dim, self.block_n, self.group
        nw, ng = D // PACK_G, D // g
        q = self.v_q[blk].to(torch.int64) & 0xFFFFFFFF           # [H, BN, nw]
        shifts = torch.arange(PACK_G, device=self.device) * 4
        val = (q.unsqueeze(-1) >> shifts) & 0xF                   # [H, BN, nw, 8]
        val = val.to(torch.int32)
        val = torch.where(val >= 8, val - 16, val).float()
        s = self.v_s[blk].float().repeat_interleave(g, dim=-1)    # [H, BN, D]
        deq = val.reshape(H, BN, D) * s
        return deq.unsqueeze(0).to(torch.float16)
