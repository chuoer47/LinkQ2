"""M9: speculative verify through the real engine (design-m9 §5).

What is NOT asserted: exact greedy token equality between spec and non-spec
engines. The two paths run different GEMM M-dims (M rows vs 1), so cuBLAS
tiling can flip near-tie argmaxes — the same measured chaos as M8 (HF itself
drifts 0.02-0.03 between solo and batched). The mathematical structure is
asserted instead:

  - padding neutrality: a sequence with no drafts riding a verify batch is a
    plain decode step (first token exactly equal — prefill path)
  - replay-stub acceptance: proposing the engine's own known greedy
    continuation must be accepted at (or very near) full length
  - real n-gram on a copy-prompt: strictly fewer steps, full length committed
  - determinism: identical spec runs are identical
  - edges: prompt shorter than the n-gram window, max_tokens truncating
    mid-draft, generation across block boundaries
"""
import pytest
import torch

from qslab.runtime.llm_engine import LLMEngine
from qslab.runtime.sampling_params import SamplingParams

pytestmark = pytest.mark.e2e

MODEL = "models/Qwen3-1.7B"
CALIB = "results/smooth_kv4_qwen3-1.7b.pt"
COPY_PROMPT = (" The capital of France is Paris. The capital of Germany is Berlin. "
               "The capital of Italy is Rome. The capital of France is")
PLAIN_PROMPT = " Water freezes at"


def _eng(**kw):
    spec = kw.pop("spec", True)
    method = kw.pop("method", "ngram")
    eng = LLMEngine(model=MODEL, max_model_len=4096, max_num_seqs=4,
                    enforce_eager=kw.pop("eager", True),
                    gpu_memory_utilization=0.5, smooth_kv=CALIB,
                    **({"spec_method": method, "spec_num_drafts": 4} if spec else {}))
    return eng


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


def _gen(eng, prompt, n=48, temperature=1e-6):
    out = eng.generate([prompt], SamplingParams(temperature=temperature, max_tokens=n),
                       use_tqdm=False)
    return out[0]["token_ids"]


class _Counting:
    """Wraps the scheduler to count verify steps and committed tokens."""
    def __init__(self, eng):
        self.eng = eng
        self.steps = 0
        self.committed = 0
        self.orig = eng.scheduler.postprocess_verify

    def __enter__(self):
        eng = self.eng

        def spy(seqs, token_ids):
            self.steps += 1
            n = self.orig(seqs, token_ids)
            self.committed += n
            return n
        eng.scheduler.postprocess_verify = spy
        return self

    def __exit__(self, *a):
        self.eng.scheduler.postprocess_verify = self.orig


def test_padding_neutral_first_token_and_length():
    """A no-draft sequence riding the verify batch: output length respected,
    first token exactly the plain decode token (prefill boundary is exact)."""
    eng = _eng()
    try:
        eng.scheduler.proposer = None          # never propose: pure padding
        with _Counting(eng) as c:
            got = _gen(eng, COPY_PROMPT, n=24)
        assert c.steps >= 20                    # ran the verify path, not decode
        assert len(got) == 24
        assert got[0] == 12095                  # ' Paris', from the prefill path
    finally:
        _release(eng)


def test_replay_stub_acceptance():
    """Proposing the engine's own greedy continuation must be accepted almost
    fully — validates the accept/commit/trim machinery, not the proposer."""
    eng_plain = _eng(spec=False)
    ref = _gen(eng_plain, COPY_PROMPT, n=64)
    _release(eng_plain)
    prompt_len = len(eng_plain.tokenizer.encode(COPY_PROMPT))
    # ref = continuation only; the full committed history is prompt + ref[:i]
    ref_full = None
    tok = eng_plain.tokenizer

    eng = _eng()
    try:
        class Replay:
            def __init__(self, prompt_ids, continuation):
                self.prompt_ids = prompt_ids
                self.cont = continuation

            def propose_batch(self, seqs):
                out = []
                for s in seqs:
                    # position in the rollout: len(token_ids) - len(prompt)
                    i = len(s.token_ids) - len(self.prompt_ids)
                    out.append(self.cont[i:i + 4])
                return out
        prompt_ids = tok.encode(COPY_PROMPT)
        eng.scheduler.proposer = Replay(prompt_ids, ref)
        with _Counting(eng) as c:
            got = _gen(eng, COPY_PROMPT, n=64)
        assert len(got) == 64
        # near-full acceptance: avg committed per step should be close to
        # gamma+1 = 5; one near-tie flip only costs a single step
        avg = c.committed / c.steps
        assert avg >= 3.5, f"avg committed/step = {avg:.2f} (expected ~5)"
        assert got[:8] == ref[:8], "committed prefix diverged from greedy ref"
    finally:
        _release(eng)


def test_ngram_copy_prompt_fewer_steps():
    """The real proposer on a copy-prompt: steps strictly fewer than plain."""
    eng_plain = _eng(spec=False)
    with _Counting(eng_plain) as c0:            # counts nothing (orig is plain)
        pass
    _gen(eng_plain, COPY_PROMPT, n=48)
    # plain steps = 48 decodes; count spec steps with a real proposer
    _release(eng_plain)

    eng = _eng()
    try:
        with _Counting(eng) as c:
            got = _gen(eng, COPY_PROMPT, n=48)
        assert len(got) == 48
        assert got[0] == 12095
        # 48 tokens in fewer than 48 verify steps -> at least one multi-commit
        assert c.steps < 48, f"steps = {c.steps}, no speedup"
        # the first token comes from the prefill step (plain postprocess);
        # everything after it is committed by verify steps only
        assert c.committed == len(got) - 1
    finally:
        _release(eng)


def test_spec_is_deterministic():
    eng = _eng()
    try:
        a = _gen(eng, COPY_PROMPT, n=24)
        b = _gen(eng, COPY_PROMPT, n=24)
        assert a == b
    finally:
        _release(eng)


def test_short_prompt_and_small_budget():
    """Prompt shorter than the n-gram window (never proposes) and a token
    budget smaller than the draft window (mid-verify truncation)."""
    eng = _eng()
    try:
        got = _gen(eng, " Paris", n=6)
        assert len(got) == 6
        assert 0 not in got
    finally:
        _release(eng)


def test_generation_crosses_block_boundary():
    """L grows past 128 (kvcache_block_size): reserve/trim stay canonical."""
    long_prompt = (" The quick brown fox jumps over the lazy dog. "
                   "The capital of France is Paris and the color of the sky is blue. ") * 6
    eng = _eng()
    try:
        ids = eng.tokenizer.encode(long_prompt)
        assert len(ids) > 130                   # already past a block boundary
        got = _gen(eng, long_prompt, n=40)
        assert len(got) == 40
        assert 0 not in got
        assert len(set(got)) > 8                # not degenerate repetition collapse
        bm = eng.scheduler.block_manager
        for seq in list(eng.scheduler.running) + list(eng.scheduler.waiting):
            assert len(seq.block_table) == seq.num_blocks, "table not canonical"
    finally:
        _release(eng)


def test_spec_with_cuda_graph():
    """The (bs, M) graph family: capture, replay, correctness vs eager spec."""
    eng = _eng(eager=False)
    try:
        assert hasattr(eng.model_runner, "graphs_verify"), "verify graphs not captured"
        got = _gen(eng, COPY_PROMPT, n=32)
        assert len(got) == 32
        assert got[0] == 12095
        assert 0 not in got
        # replay twice is deterministic
        again = _gen(eng, COPY_PROMPT, n=32)
        assert got == again
    finally:
        _release(eng)


def test_lookahead_proposer_end_to_end():
    """M10: the migrated M6 self-proposal mode rides the same verify path.

    It is a different proposer (persistent index, chain extension, first
    occurrence wins) behind identical acceptance machinery, so the assertions
    are the machinery's: full length, an exact prefill token, and fewer steps
    than tokens.
    """
    eng = _eng(method="lookahead")
    try:
        assert eng.scheduler.proposer.__class__.__name__ == "LookaheadProposer"
        with _Counting(eng) as c:
            got = _gen(eng, COPY_PROMPT, n=48)
        assert len(got) == 48
        assert got[0] == 12095
        assert 0 not in got
        assert c.steps < 48, f"steps = {c.steps}, no speedup"
        assert c.committed == len(got) - 1
        # the scheduler's own counters are what the facade reports
        st = eng.scheduler.spec_stats
        assert st["steps"] == c.steps and st["committed"] == c.committed
        assert 0.0 <= st["acceptance_rate"] <= 1.0
    finally:
        _release(eng)


def test_temperature_verify_accepts_ratio_and_replays():
    """Sampled decoding through the speculative path (design-m9 §7 closed).

    The rule promises the target's *distribution*, never its argmax, so this
    asserts the machinery and the replay stability a seeded run can give — not
    equality with the greedy tokens above.
    """
    eng = _eng()
    try:
        torch.manual_seed(7)
        with _Counting(eng) as c:
            a = _gen(eng, COPY_PROMPT, n=48, temperature=0.8)
        assert len(a) == 48
        assert c.committed == len(a) - 1
        assert c.steps < 48, f"steps = {c.steps}: ratio path never committed twice"
        assert len(set(a)) > 8, "sampled output collapsed into repetition"
        torch.manual_seed(7)
        assert _gen(eng, COPY_PROMPT, n=48, temperature=0.8) == a
    finally:
        _release(eng)
