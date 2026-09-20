"""WikiText-2 PPL through the paged runtime's real decode path.

Protocol (docs/03): sliding window 1024, stride 512, WikiText-2 raw test.

Why teacher forcing: the int4 cache is only read during decode — a
prefill-only pass never touches it, because flash-attn consumes the freshly
computed fp16 K/V. So the loop prefills a context and then feeds the
ground-truth token one step at a time, scoring each prediction. That is the
regime the 4-bit cache actually affects, and it is the same code path real
generation uses (block allocation, slot mapping, context lengths all advance
normally) — only the sampled token is replaced by the ground truth.

Sequences are retired by giving them a max_tokens budget that the forced
tokens exhaust, so the block manager releases blocks through its normal path
rather than being patched.
"""
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from qslab.runtime.engine.llm_engine import LLMEngine
from qslab.runtime.sampling_params import SamplingParams

WINDOW = 1024
STRIDE = 512
N_DOCS = int(os.environ.get("N_DOCS", "8"))
N_CTX = int(os.environ.get("N_CTX", "128"))
N_DEC = int(os.environ.get("N_DEC", "128"))


def load_texts():
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return [t for t in ds["text"] if len(t.strip()) > 200][:N_DOCS]


def score_window(eng, cap, chunk):
    """Prefill chunk[:N_CTX], then force-decode the rest, scoring each step."""
    nll, n = 0.0, 0
    prompt = chunk[:N_CTX]
    budget = N_DEC
    eng.add_request(prompt, SamplingParams(temperature=1e-6, max_tokens=budget))
    seqs, pre = eng.scheduler.schedule()
    if not pre:
        return 0.0, 0
    eng.model_runner.call("run", seqs, pre)
    eng.scheduler.postprocess(seqs, [prompt[-1]], pre)

    for i in range(N_CTX, min(N_CTX + budget, len(chunk))):
        seqs, pre = eng.scheduler.schedule()
        if pre or not seqs:
            break
        eng.model_runner.call("run", seqs, pre)
        lg = cap["logits"][0]
        nll -= torch.log_softmax(lg, dim=-1)[chunk[i]].item()
        n += 1
        eng.scheduler.postprocess(seqs, [chunk[i]], pre)   # force the token
    return nll, n


def ppl(eng, tok, texts):
    mr = eng.model_runner
    cap = {}
    orig = mr.run_model

    def spy(input_ids, positions, is_prefill):
        logits = orig(input_ids, positions, is_prefill)
        cap["logits"] = logits.detach().float()
        return logits

    mr.run_model = spy
    nll_sum, n_tok = 0.0, 0
    try:
        for text in texts:
            ids = tok.encode(text)
            if len(ids) < N_CTX + 8:
                continue
            # one window per document, from the document start
            chunk = ids[: min(WINDOW, len(ids))]
            a, b = score_window(eng, cap, chunk)
            nll_sum += a
            n_tok += b
    finally:
        mr.run_model = orig
    return math.exp(nll_sum / max(n_tok, 1)), n_tok


if __name__ == "__main__":
    from adapters.tokenizer import QwenTokenizerAdapter

    tok = QwenTokenizerAdapter(os.environ["MODEL"])
    texts = load_texts()
    kw = {}
    if os.environ.get("CALIB"):
        kw["smooth_kv"] = os.environ["CALIB"]
    if os.environ.get("W4"):
        kw["w4"] = os.environ["W4"]
    eng = LLMEngine(model=os.environ["MODEL"], max_model_len=4096,
                    max_num_seqs=4, enforce_eager=True,
                    gpu_memory_utilization=float(os.environ.get("UTIL", "0.5")),
                    **kw)
    val, n = ppl(eng, tok, texts)
    print(f"RESULT {os.environ.get('TAG','run')}: PPL {val:.4f}  ({n} tokens)")
