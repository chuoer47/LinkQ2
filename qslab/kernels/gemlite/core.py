"""Autotune cache consumed by the upstream kernel pruners."""
import json
from pathlib import Path

GEMLITE_TRITON_CONFIG_CACHE = {}


def load_config(path):
    GEMLITE_TRITON_CONFIG_CACHE.update(json.loads(Path(path).read_text()))


def save_config(path):
    from .ops import ROUTES
    configs = dict(GEMLITE_TRITON_CONFIG_CACHE)
    for name, route in ROUTES.items():
        rows = configs.setdefault(name, {})
        for key, config in route.kernel.cache.items():
            rows[str(tuple(key[:6]))] = config.all_kwargs()
    Path(path).write_text(json.dumps(configs, indent=2))
