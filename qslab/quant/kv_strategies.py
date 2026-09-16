"""L1: KVCacheStrategy — pluggable KV cache precision/placement strategy.

Previously the engine chose the cache class with inline if/elif over a
``kv_mode`` string (qslab/engine/core.py). That decision now lives here:
each strategy knows how to build per-layer caches and how to describe its
memory footprint.

The ``paged`` slot is RESERVED for M7 (quantized paged attention): a future
``KV4PagedStrategy`` will implement the same interface over block-mapped
storage so the engine needs no change.
"""
from __future__ import annotations

from typing import Protocol

from qslab.registry import Registry

KV_STRATEGIES = Registry("kv cache strategy")


class KVCacheStrategy(Protocol):
    """Builds per-layer caches for a model shape."""

    name: str

    def build(self, layer_idx: int, batch: int, num_kv_heads: int,
              head_dim: int, max_len: int, device: str):
        """Return a cache instance for one decoder layer."""
        ...

    @property
    def is_quantized(self) -> bool: ...
