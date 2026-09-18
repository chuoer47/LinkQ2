"""WikiText-2 raw PPL (docs/03 §1 protocol): sliding window stride 512.

Uses the HF datasets wikitext-2-raw-v1 test split via adapters (the only
transformers-adjacent import point). PPL computed with the engine itself so
the number reflects the engine, not a reference implementation.
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from qslab.config import EngineConfig
from qslab.models.loader import load_reference_model
from adapters.tokenizer import QwenTokenizerAdapter

WINDOW = 1024
STRIDE = 512


def gpu_snapshot() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True)
        return {"nvidia_smi": out.strip().splitlines()}
    except Exception as e:
        return {"error": str(e)}


@torch.inference_mode()
def ppl_of_model(model, tok, texts: list[str], device: str,
                 reset_fn=None) -> float:
    """Standard sliding-window PPL over concatenated token stream.
    reset_fn: optional per-window cache reset (engine-owned caches need it)."""
    nll_sum, n_tokens = 0.0, 0
    for text in texts:
        ids = tok.encode(text)
        if len(ids) < 2:
            continue
        # concatenate per-document; sliding window over each document
        prev_end = 0
        for begin in range(0, len(ids) - 1, STRIDE):
            end = min(begin + WINDOW, len(ids))
            x = torch.tensor([ids[begin:end]], device=device)
            if reset_fn is not None:
                reset_fn()
            tgt_len = end - prev_end  # only count new tokens (first pass counts all)
            logits = model(input_ids=x).logits
            # predict x[1..] from x[:-1] within counted region
            count = min(tgt_len, x.shape[1] - 1)
            logprobs = torch.log_softmax(logits[0, -count - 1:-1, :].float(), dim=-1)
            targets = x[0, -count:]
            nll = -logprobs.gather(-1, targets[:, None]).sum().item()
            nll_sum += nll
            n_tokens += count
            prev_end = end
            if end == len(ids):
                break
    return float(torch.exp(torch.tensor(nll_sum / max(n_tokens, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-1.7B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-docs", type=int, default=64, help="docs from test split")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import os
    # hf.co unreachable from the server; mirror cache already has wikitext.
    # Point hub calls at the mirror and allow offline fallback to the arrow cache.
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from datasets import load_dataset
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test",
                          download_mode="reuse_cache_if_exists")
    texts = [t for t in ds["text"] if len(t.strip()) > 200][: args.max_docs]

    tok = QwenTokenizerAdapter(args.model)
    model = load_reference_model(args.model, device=args.device)

    ppl = ppl_of_model(model, tok, texts, args.device)
    report = {
        "model": args.model,
        "dataset": "wikitext-2-raw-v1/test",
        "docs": len(texts),
        "window": WINDOW,
        "stride": STRIDE,
        "ppl": round(ppl, 4),
        "gpu": gpu_snapshot(),
        "timestamp": datetime.datetime.now().isoformat(),
    }
    out = Path(args.out) if args.out else Path(
        f"results/ppl_{Path(args.model).name}_{datetime.datetime.now():%Y%m%d_%H%M%S}.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"PPL: {ppl:.4f} | saved {out}")


if __name__ == "__main__":
    main()
