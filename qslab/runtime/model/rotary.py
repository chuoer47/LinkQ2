"""qslab runtime rotary embedding — vendored from nano-vllm (MIT), unchanged
except for the import paths and single-GPU simplifications. YaRN support is
qslab's own: `rope_scaling` used to be accepted by Qwen3Attention and dropped,
so nothing in the runtime could address a position past the native ceiling."""
from __future__ import annotations

import math

import torch
from torch import nn


def apply_rotary_emb(x, cos, sin):
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


def _yarn_inv_freq(dim: int, base: float, factor: float,
                   original_max_position: int, beta_fast: float = 32.0,
                   beta_slow: float = 1.0, truncate: bool = True):
    """Blend the extrapolated and interpolated inverse frequencies per dim.

    Transcribed from transformers' _compute_yarn_parameters so the two agree
    numerically — tests/runtime/model/test_rotary_yarn.py locks that against the library
    rather than against this file.
    """
    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
    inv_extrapolation = 1.0 / pos_freqs
    inv_interpolation = 1.0 / (factor * pos_freqs)

    def correction_dim(num_rotations):
        return (dim * math.log(original_max_position / (num_rotations * 2 * math.pi))
                / (2 * math.log(base)))

    low, high = correction_dim(beta_fast), correction_dim(beta_slow)
    if truncate:
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high = low + 0.001          # avoid the ramp's singularity

    half = dim // 2
    ramp = torch.clamp((torch.arange(half, dtype=torch.float32) - low) / (high - low), 0, 1)
    extrap_share = 1 - ramp
    return (inv_interpolation * (1 - extrap_share)
            + inv_extrapolation * extrap_share)


def _yarn_attention_factor(scaling: dict, factor: float) -> float:
    """mscale: how much to widen the attention logits' denominator. HF's rule."""
    given = scaling.get("attention_factor")
    if given is not None:
        return float(given)
    if factor <= 1:
        return 1.0
    mscale, mscale_all_dim = scaling.get("mscale"), scaling.get("mscale_all_dim")
    if mscale and mscale_all_dim:
        return float((0.1 * mscale * math.log(factor) + 1.0)
                     / (0.1 * mscale_all_dim * math.log(factor) + 1.0))
    return 0.1 * math.log(factor) + 1.0


class RotaryEmbedding(nn.Module):
    def __init__(self, head_size: int, rotary_dim: int,
                 max_position_embeddings: int, base: float,
                 scaling: dict | None = None):
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        rope_type = (scaling or {}).get("rope_type", (scaling or {}).get("type"))
        if scaling and rope_type != "yarn":
            # a non-empty dict that does not name a type we implement is an
            # error, not a request for the native rope — transformers raises on
            # an unknown type too, and a silent fallback here would be a
            # length-extension run that measured nothing.
            raise NotImplementedError(
                f"rope_type {rope_type!r}: the runtime only implements 'yarn'")

        if rope_type == "yarn":
            factor = float(scaling["factor"])
            original = int(scaling.get("original_max_position_embeddings")
                           or max_position_embeddings)
            inv_freq = _yarn_inv_freq(
                rotary_dim, base, factor, original,
                beta_fast=float(scaling.get("beta_fast") or 32),
                beta_slow=float(scaling.get("beta_slow") or 1),
                truncate=bool(scaling.get("truncate", True)))
            self.attention_scaling = _yarn_attention_factor(scaling, factor)
        else:
            inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2,
                                                    dtype=torch.float) / rotary_dim))
            # multiplying by exactly 1.0 is bit-exact, so the unscaled cache is
            # the cache every other milestone measured
            self.attention_scaling = 1.0

        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * self.attention_scaling
        sin = freqs.sin() * self.attention_scaling
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(self, positions, query, key):
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


#: one instance per distinct rope setup — all 36 decoder layers share it, as
#: the previous lru_cache(1) guaranteed. A dict is unhashable, so the key is the
#: sorted items.
_CACHE: dict[tuple, RotaryEmbedding] = {}


def get_rope(head_size: int, rotary_dim: int, max_position: int, base: float,
             scaling: dict | None = None):
    key = (head_size, rotary_dim, max_position, base,
           tuple(sorted((scaling or {}).items())))
    rope = _CACHE.get(key)
    if rope is None:
        rope = _CACHE[key] = RotaryEmbedding(head_size, rotary_dim, max_position,
                                            base, scaling)
    return rope
