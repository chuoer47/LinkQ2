"""Shared pytest fixtures and markers for qslab tests.

Test tiers:
  - fast (default): pure-CPU logic (sampling math, packing, registries)
  - gpu: needs a CUDA device (kernel numerics, cache roundtrip)
  - e2e: needs model weights on disk (oracle alignment) — skipped if missing

Run:  pytest tests/ -v                 # everything runnable here
      pytest tests/ -m "not e2e"       # skip model-dependent tests
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# model/checkpoint paths used by e2e tests
MODELS = REPO / "models"
SMALL_MODEL = MODELS / "Qwen3-1.7B"
SMALL_W4 = MODELS / "Qwen3-1.7B-qslab-w4-awq2"
DRAFT_MODEL = MODELS / "Qwen3-0.6B"


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires a CUDA device")
    config.addinivalue_line("markers", "e2e: requires model weights on disk")


@pytest.fixture(scope="session")
def has_cuda() -> bool:
    import torch
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def has_small_model() -> bool:
    return SMALL_MODEL.exists() and SMALL_W4.exists()


@pytest.fixture(autouse=True)
def _release_gpu_between_modules(request):
    """Engine-holding test modules take turns on one 24 GB card.

    Each module is torn down before the next one builds its engine, because
    two 1.7B engines plus their KV pools do not co-reside. Threading the
    cleanup through a module-scoped autouse fixture keeps it out of the
    individual tests.
    """
    yield
    if request.node.get_closest_marker("e2e") is None:
        return
    import gc
    try:
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:
        pass
