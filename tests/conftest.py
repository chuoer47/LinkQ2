"""Shared pytest fixtures and markers for qslab tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires a CUDA device")
    config.addinivalue_line("markers", "e2e: requires model weights on disk")


@pytest.fixture(autouse=True)
def _release_gpu_between_modules(request):
    """Engine-holding test modules take turns on the single GPU."""
    # Two 1.7B engines plus their KV pools do not co-reside, so each module is torn down
    #   before the next one builds its engine.
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
