"""End-to-end tests: engine vs HF oracle, spec losslessness (need weights)."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.e2e

from conftest import DRAFT_MODEL, SMALL_MODEL, SMALL_W4  # noqa: E402

pytestmark = [pytest.mark.e2e, pytest.mark.gpu]


@pytest.fixture(autouse=True)
def _needs_assets():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    if not (SMALL_MODEL.exists() and SMALL_W4.exists()):
        pytest.skip("model weights not on disk")


class TestEngineOracle:
    def test_fp16_engine_matches_hf_greedy(self):
        from qslab.api.llm import LLM, SamplingParams

        llm = LLM(str(SMALL_MODEL))
        prompt = " The capital of France is"
        got = llm.generate(prompt, SamplingParams(max_tokens=24))["token_ids"]

        # HF oracle
        from qslab.models.loader import load_reference_model
        from adapters.tokenizer import QwenTokenizerAdapter
        tok = QwenTokenizerAdapter(SMALL_MODEL)
        ref = load_reference_model(SMALL_MODEL)
        ids = tok.encode(prompt)
        with torch.inference_mode():
            out = ref.generate(torch.tensor([ids], device="cuda:0"),
                               max_new_tokens=24, do_sample=False)
        assert got == out[0][len(ids):].tolist()


class TestSpecLosslessness:
    @pytest.mark.parametrize("mode", ["chained", "lookahead", "dynamic"])
    def test_speculation_is_lossless(self, mode):
        """Every mode must reproduce target-only greedy exactly."""
        from qslab.api.llm import LLM, SamplingParams

        prompt = " The capital of France is"
        params = SamplingParams(max_tokens=24)

        base = LLM(str(SMALL_MODEL), w4=str(SMALL_W4))
        ref_tokens = base.generate(prompt, params)["token_ids"]

        spec = LLM(str(SMALL_MODEL), w4=str(SMALL_W4), draft=str(DRAFT_MODEL),
                   spec_mode=mode, spec_gamma=4)
        got = spec.generate(prompt, params)["token_ids"]
        assert got == ref_tokens, f"{mode} broke losslessness"
