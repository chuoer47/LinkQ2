"""M8: runtime-level properties — CUDA Graph, batching, and the batch-size
sensitivity of greedy decoding.

Background for the last test: batching changes the M dimension of every dense
GEMM, so cuBLAS picks different tiling and the reduction order — and hence the
last bits of each matmul — changes. Greedy decoding can then flip at a
near-tie. This is not specific to qslab: transformers shows the same
magnitude of drift (0.02-0.03 in logits) for the same prompt batched vs
alone. What must hold is that the *batched* result matches a reference
*batched* result, and that the divergence does not depend on other sequences'
contents (which would indicate cache aliasing).
"""
import pytest
import torch

from qslab.runtime.llm_engine import LLMEngine
from qslab.runtime.sampling_params import SamplingParams
from flash_attn import flash_attn_varlen_func

pytestmark = pytest.mark.e2e

MODEL = "models/Qwen3-1.7B"
CALIB = "results/smooth_kv4_qwen3-1.7b.pt"
PROMPT = " The capital of France is"
ORACLE16 = [12095, 13, 576, 6722, 315, 9856, 374, 19846, 13, 576, 6722, 315,
            15344, 374, 21718, 13]


def free_engine(eng):
    """Drop the model, the KV pool and the graphs, then hand the memory back.

    Modules in this file take turns on one card, so each engine must be gone
    before the next one is built; `del` alone leaves the tensors alive because
    the attention layers still hold references to them.
    """
    import gc
    runner = eng.model_runner
    for layer in runner.model.model.layers:
        a = layer.self_attn.attn
        a.k_cache = a.v_cache = None
        a.lam = a.lam_q = None
    runner.model = None
    runner.kv_cache = None
    if hasattr(runner, "graphs"):
        runner.graphs = None
        runner.graph_pool = None
    del eng.model_runner
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def build(**kw):
    # 0.35 keeps a second engine from being needed at once; two 1.7B engines
    # plus their KV pools do not fit a 24 GB card.
    return LLMEngine(model=MODEL, max_model_len=4096, max_num_seqs=8,
                     gpu_memory_utilization=0.35, smooth_kv=CALIB, **kw)


def gen(eng, prompts, max_tokens=16):
    outs = eng.generate(prompts, SamplingParams(temperature=1e-6,
                                                max_tokens=max_tokens),
                        use_tqdm=False)
    return [o["token_ids"] for o in outs]


@pytest.fixture(scope="module")
def graph_engine():
    eng = build(enforce_eager=False)
    yield eng
    free_engine(eng)


def test_cuda_graph_captured_and_lossless(graph_engine):
    """The int4 paged path must be graph-capturable.

    This is the entire reason K carries a frozen per-channel scale instead of
    a per-token one: a scale recomputed as tokens arrive would make the write
    address data-dependent, and graph capture requires it to be a pure
    function of slot_mapping.
    """
    graphs = getattr(graph_engine.model_runner, "graphs", None)
    assert graphs, "capture_cudagraph produced no graphs"
    assert set(graphs) >= {1, 2, 4, 8}, f"unexpected batch buckets {sorted(graphs)}"
    assert gen(graph_engine, [PROMPT])[0] == ORACLE16


def test_batch_composition_does_not_leak(graph_engine):
    """A sequence's output must not depend on its companions' contents."""
    mates = [" The largest city in Japan is", " Water freezes at",
             " 2 * 3 =", " The opposite of hot is"]
    reference = gen(graph_engine, [PROMPT, mates[0]])[0]
    for mate in mates[1:]:
        assert gen(graph_engine, [PROMPT, mate])[0] == reference, \
            f"output changed when batched with {mate!r}"


def test_identical_prompts_agree_in_batch(graph_engine):
    outs = gen(graph_engine, [PROMPT] * 4)
    assert all(o == outs[0] for o in outs), "identical prompts diverged in a batch"


def test_varlen_prefill_is_batch_invariant():
    """flash-attn varlen gives identical output regardless of what else is
    packed into the same call — so the prefill cannot be the source of any
    batch-size-dependent drift."""
    H_Q, H_KV, D = 16, 8, 128
    g = torch.Generator(device="cuda").manual_seed(0)
    qa = torch.randn(6, H_Q, D, device="cuda", generator=g).half()
    ka = torch.randn(6, H_KV, D, device="cuda", generator=g).half()
    va = torch.randn(6, H_KV, D, device="cuda", generator=g).half()

    def call(parts, lens):
        q = torch.cat([p[0] for p in parts], 0)
        k = torch.cat([p[1] for p in parts], 0)
        v = torch.cat([p[2] for p in parts], 0)
        cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(lens), 0).tolist()),
                          dtype=torch.int32, device="cuda")
        return flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                                      max_seqlen_q=max(lens), max_seqlen_k=max(lens),
                                      softmax_scale=D ** -0.5, causal=True)

    solo = call([(qa, ka, va)], [6])
    for extra in (6, 60, 120, 248):
        g2 = torch.Generator(device="cuda").manual_seed(1)
        qb = torch.randn(extra, H_Q, D, device="cuda", generator=g2).half()
        kb = torch.randn(extra, H_KV, D, device="cuda", generator=g2).half()
        vb = torch.randn(extra, H_KV, D, device="cuda", generator=g2).half()
        packed = call([(qa, ka, va), (qb, kb, vb)], [6, extra])
        assert torch.equal(solo, packed[:6]), f"varlen changed with a {extra}-token mate"
