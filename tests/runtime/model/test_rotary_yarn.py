"""YaRN lock: the runtime's rope scaling must agree with transformers."""
from __future__ import annotations

import math
import os

import pytest
import torch

from qslab.runtime.model.rotary import (RotaryEmbedding, _yarn_attention_factor,
                                  _yarn_inv_freq, get_rope)

pytest.importorskip("transformers")
from transformers import Qwen3Config  # noqa: E402
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding  # noqa: E402

# Qwen3-8B's rope geometry
HEAD_DIM = 128
BASE = 1_000_000.0
NATIVE_MAX = 40_960
FACTOR = 3.2                       # 40960 * 3.2 = 131072
YARN = {"rope_type": "yarn", "factor": FACTOR,
        "original_max_position_embeddings": NATIVE_MAX}
EXTENDED_MAX = int(NATIVE_MAX * FACTOR)
# inside the native window, at the native ceiling, and at the extended ceiling
POSITIONS = [0, 1, 7, 400, 5000, NATIVE_MAX - 1, EXTENDED_MAX - 1]


def _hf_reference(scaling: dict, max_position: int = EXTENDED_MAX):
    cfg = Qwen3Config(hidden_size=4096, num_attention_heads=32,
                      num_key_value_heads=8, head_dim=HEAD_DIM,
                      max_position_embeddings=max_position, rope_theta=BASE,
                      rope_scaling=dict(scaling))
    return Qwen3RotaryEmbedding(cfg)


def _hf_cos_sin(ref: Qwen3RotaryEmbedding, positions: list[int]):
    """HF returns (1, n, head_dim) with the half-width freqs duplicated."""
    pos = torch.tensor([positions])
    dummy = torch.zeros(1, len(positions), 1)
    cos, sin = ref(dummy, pos)
    half = HEAD_DIM // 2
    return cos[0, :, :half], sin[0, :, :half]


class TestYarnAgainstTransformers:
    def test_inv_freq_matches_library_exactly(self):
        ref = _hf_reference(YARN)
        torch.testing.assert_close(_yarn_inv_freq(HEAD_DIM, BASE, FACTOR, NATIVE_MAX),
                                   ref.inv_freq, rtol=0, atol=1e-9)

    def test_attention_factor_matches_library(self):
        ref = _hf_reference(YARN)
        ours = _yarn_attention_factor(YARN, FACTOR)
        assert ours == pytest.approx(ref.attention_scaling, rel=0, abs=0)
        # ... and it is HF's default rule, not a value we copy-pasted
        assert ours == 0.1 * math.log(FACTOR) + 1.0

    def test_explicit_attention_factor_wins(self):
        scaling = dict(YARN, attention_factor=1.5)
        assert _yarn_attention_factor(scaling, FACTOR) == 1.5

    def test_cos_sin_cache_matches_library_position_by_position(self):
        ref = _hf_reference(YARN)
        ours = RotaryEmbedding(HEAD_DIM, HEAD_DIM, EXTENDED_MAX, BASE, YARN)
        cos, sin = _hf_cos_sin(ref, POSITIONS)
        row = ours.cos_sin_cache[torch.tensor(POSITIONS), 0]
        half = HEAD_DIM // 2
        torch.testing.assert_close(row[:, :half], cos, rtol=0, atol=0)
        torch.testing.assert_close(row[:, half:], sin, rtol=0, atol=0)

    def test_scaling_actually_changes_the_rope(self):
        """Guards against the silent-fallback failure mode: rope_scaling is a dict threaded
        through several layers."""
        yarn = RotaryEmbedding(HEAD_DIM, HEAD_DIM, EXTENDED_MAX, BASE, YARN)
        native = RotaryEmbedding(HEAD_DIM, HEAD_DIM, EXTENDED_MAX, BASE, None)
        assert not torch.equal(yarn.cos_sin_cache[:NATIVE_MAX],
                               native.cos_sin_cache[:NATIVE_MAX])

    def test_legacy_type_key_is_accepted(self):
        legacy = {"type": "yarn", "factor": FACTOR,
                  "original_max_position_embeddings": NATIVE_MAX}
        ours = RotaryEmbedding(HEAD_DIM, HEAD_DIM, EXTENDED_MAX, BASE, legacy)
        ref = _hf_reference(legacy)
        cos, _ = _hf_cos_sin(ref, POSITIONS)
        torch.testing.assert_close(
            ours.cos_sin_cache[torch.tensor(POSITIONS), 0, :HEAD_DIM // 2],
            cos, rtol=0, atol=0)

    def test_other_scaling_types_are_refused_not_ignored(self):
        with pytest.raises(NotImplementedError):
            RotaryEmbedding(HEAD_DIM, HEAD_DIM, EXTENDED_MAX, BASE,
                            {"rope_type": "dynamic", "factor": FACTOR})
        # a dict that names no type at all (misspelled key) must not quietly
        # become the native rope either
        with pytest.raises(NotImplementedError):
            RotaryEmbedding(HEAD_DIM, HEAD_DIM, EXTENDED_MAX, BASE,
                            {"factor": FACTOR})


class TestNativePathIsUntouched:
    def test_unscaled_cache_is_bit_identical_to_the_vendored_one(self):
        """Recompute the pre-YaRN cache from its own source and compare."""
        inv_freq = 1.0 / (BASE ** (torch.arange(0, HEAD_DIM, 2,
                                                dtype=torch.float) / HEAD_DIM))
        t = torch.arange(2048, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        expected = torch.cat((freqs.cos(), freqs.sin()), dim=-1).unsqueeze_(1)
        ours = RotaryEmbedding(HEAD_DIM, HEAD_DIM, 2048, BASE, None)
        assert ours.attention_scaling == 1.0
        assert torch.equal(ours.cos_sin_cache, expected)

    def test_get_rope_shares_one_instance_per_setup(self):
        """36 decoder layers must not build 36 caches (nor 36 different ones)."""
        a, b = (get_rope(HEAD_DIM, HEAD_DIM, 2048, BASE) for _ in range(2))
        assert a is b
        n = get_rope(HEAD_DIM, HEAD_DIM, 2048, BASE,
                     dict(YARN, factor=2.0))
        assert n is not a
        assert get_rope(HEAD_DIM, HEAD_DIM, 2048, BASE,
                        dict(YARN, factor=2.0)) is n


MODEL = "models/Qwen3-1.7B"


@pytest.mark.skipif(not os.path.isdir(MODEL), reason="no checkpoint")
@pytest.mark.e2e
class TestConfigOpensTheCeiling:
    """The rope math is useless if config.py's clamp still pins positions to the native
    ceiling."""

    def test_native_config_still_clamps(self):
        from qslab.runtime.config import Config
        cfg = Config(MODEL, max_model_len=EXTENDED_MAX)
        assert cfg.rope_scaling is None
        assert cfg.max_model_len == cfg.hf_config.max_position_embeddings

    def test_yarn_config_raises_it_to_the_extended_length(self):
        from qslab.runtime.config import Config
        native = int(Config(MODEL).hf_config.max_position_embeddings)
        cfg = Config(MODEL, max_model_len=native * 2,
                     rope_scaling={"rope_type": "yarn", "factor": 2.0})
        assert cfg.hf_config.max_position_embeddings == native * 2
        assert cfg.max_model_len == native * 2          # not clamped back down
        # the rope dict the model will hand to get_rope, and its origin
        assert cfg.rope_scaling["original_max_position_embeddings"] == native
        assert cfg.hf_config.rope_scaling == cfg.rope_scaling

    def test_shrinking_is_refused(self):
        from qslab.runtime.config import Config
        with pytest.raises(AssertionError):
            Config(MODEL, rope_scaling={"rope_type": "yarn", "factor": 0.5})
