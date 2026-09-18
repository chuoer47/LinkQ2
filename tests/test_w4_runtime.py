"""W4 packed weights driving the paged runtime.

The packed checkpoints are keyed by HF module name and store the AWQ input
scale per logical module, which is why the runtime model keeps q/k/v and
gate/up separate instead of fusing them (see qslab/runtime/qwen3.py).
"""
import pytest
import torch
from qslab.runtime.llm_engine import LLMEngine
from qslab.runtime.sampling_params import SamplingParams

pytestmark = pytest.mark.e2e

MODEL = "models/Qwen3-1.7B"
W4 = "models/Qwen3-1.7B-qslab-w4-awq2"
CALIB = "results/smooth_kv4_qwen3-1.7b.pt"
PROMPT = " The capital of France is"
ORACLE = [12095, 13, 576, 6722, 315, 9856, 374, 19846, 13, 576, 6722, 315,
          15344, 374, 21718, 13]


def _engine(**kw):
    return LLMEngine(model=MODEL, max_model_len=4096, max_num_seqs=8,
                     enforce_eager=True, gpu_memory_utilization=0.4,
                     smooth_kv=CALIB, **kw)


def _release(eng):
    import gc
    for layer in eng.model_runner.model.model.layers:
        a = layer.self_attn.attn
        a.k_cache = a.v_cache = None
    eng.model_runner.model = None
    eng.model_runner.kv_cache = None
    del eng.model_runner
    gc.collect()
    torch.cuda.empty_cache()


def test_w4_runtime_matches_fp16_reference():
    """The whole point of the integration: the packed W4 weights drive the
    paged runtime and still reproduce the fp16 oracle.

    Only a 12-token prefix is asserted: the W4 logits sit ~4.0 from fp16
    (measured, same order for v1 and Marlin backends, which agree with each
    other to 0.047), so a near-tie around token 12 flips with any numeric
    perturbation — the greedy-chaos criterion M8 already retired. The stable
    prefix still catches any wiring/kernel regression (garbage, wrong
    weights, broken AWQ fold) immediately."""
    eng = _engine(w4=W4)
    try:
        out = eng.generate([PROMPT], SamplingParams(temperature=1e-6, max_tokens=16),
                           use_tqdm=False)
        got = out[0]["token_ids"]
        assert got[:12] == ORACLE[:12], f"prefix differs: {got[:12]}"
        assert 0 not in got
        assert len(got) == 16
    finally:
        _release(eng)


def test_w4_swaps_every_quantized_linear():
    """Embeddings, norms and an untied lm_head stay fp16; everything the
    checkpoint quantized must be replaced."""
    from qslab.models.w4linear import W4Linear
    from qslab.quant.packfmt import load_qslab_w4
    from qslab.runtime.primitives import Linear as RuntimeLinear

    cfg, _, _ = load_qslab_w4(W4)
    expected = len(cfg["quantized_layers"])

    eng = _engine(w4=W4)
    try:
        model = eng.model_runner.model
        n_w4 = sum(1 for m in model.modules() if isinstance(m, W4Linear))
        n_left = sum(1 for m in model.modules() if isinstance(m, RuntimeLinear))
        assert n_w4 == expected, f"swapped {n_w4}, checkpoint has {expected}"
        # every projection is quantized, so no plain Linear should remain;
        # embeddings, norms and the softmax head stay fp16
        assert n_left == 0, f"{n_left} unswapped Linear layers remain"
        assert model.model.embed_tokens.weight.dtype == torch.float16
        assert model.lm_head.weight.dtype == torch.float16
    finally:
        _release(eng)
