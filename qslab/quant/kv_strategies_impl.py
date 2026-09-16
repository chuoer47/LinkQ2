"""L1 concrete KV cache strategies: fp16 / kv8 / kv4 (+ reserved paged).

Includes the per-layer fp16 override from a kv4 plan (the "first N layers
stay fp16" policy measured in M2).
"""
from __future__ import annotations

import json

from qslab.quant.cache.kv_cache import FP16KVCache, KV4Cache, KV8Cache
from qslab.quant.kv_strategies import KV_STRATEGIES


class _StrategyBase:
    #: class from qslab.quant.cache used for quantized layers
    cache_cls = None
    name = "base"
    is_quantized = False

    def __init__(self, group: int = 64, fp16_layers: set[int] | None = None):
        self.group = group
        self.fp16_layers = fp16_layers or set()

    def build(self, layer_idx: int, batch: int, num_kv_heads: int,
              head_dim: int, max_len: int, device: str):
        cls = FP16KVCache if (not self.is_quantized or layer_idx in self.fp16_layers) \
            else self.cache_cls
        kwargs = {} if cls is FP16KVCache else {"group": self.group}
        return cls(batch=batch, num_kv_heads=num_kv_heads, head_dim=head_dim,
                   max_len=max_len, device=device, **kwargs)


@KV_STRATEGIES.register("fp16")
class FP16Strategy(_StrategyBase):
    name = "fp16"
    is_quantized = False


@KV_STRATEGIES.register("kv8")
class KV8Strategy(_StrategyBase):
    name = "kv8"
    is_quantized = True
    cache_cls = KV8Cache


@KV_STRATEGIES.register("kv4")
class KV4Strategy(_StrategyBase):
    name = "kv4"
    is_quantized = True
    cache_cls = KV4Cache


@KV_STRATEGIES.register("kv4.plan")
class KV4PlanStrategy(KV4Strategy):
    """kv4 with the layer list kept at fp16, loaded from a kv4_plan.json."""

    name = "kv4.plan"

    @classmethod
    def from_plan_file(cls, path: str) -> "KV4PlanStrategy":
        plan = json.loads(open(path).read())
        return cls(group=plan.get("group_size", 64),
                   fp16_layers=set(plan.get("kv_fp16_layers", [])))


def build_strategy(kv_mode: str, kv_plan_path: str | None = None,
                   group: int = 64) -> _StrategyBase:
    """Factory: 'kv4' + plan file -> KV4PlanStrategy, else plain strategy."""
    if kv_mode == "kv4" and kv_plan_path:
        return KV4PlanStrategy.from_plan_file(kv_plan_path)
    cls = KV_STRATEGIES.get(kv_mode)
    return cls(group=group)
