"""M2-S4: PPL comparison across KV precisions (fp16 / kv8 / kv4) + memory.

Uses the engine itself (kv_mode) on the FP16 model — isolates KV quantization
effect. Reuses bench_ppl's sliding-window PPL but through the engine path.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from qslab.config import EngineConfig
from qslab.engine import QslabEngine
from adapters.tokenizer import QwenTokenizerAdapter

WINDOW = 1024
STRIDE = 512


@torch.inference_mode()
def ppl_engine(eng, tok, texts):
    nll_sum, n_tokens = 0.0, 0
    for text in texts:
        ids = tok.encode(text)
        if len(ids) < 2:
            continue
        prev_end = 0
        for begin in range(0, len(ids) - 1, STRIDE):
            end = min(begin + WINDOW, len(ids))
            chunk = ids[begin:end]
            eng.reset_cache()
            # full forward through the engine's model (prefill() returns only
            # last-position logits; PPL needs the whole window)
            x = torch.tensor([chunk], device=eng.device)
            logits3 = eng.model(input_ids=x, use_cache=False).logits
            count = min(end - prev_end, len(chunk) - 1)
            logprobs = torch.log_softmax(logits3[0, -count - 1:-1, :].float(), dim=-1)
            targets = torch.tensor(chunk[-count:], device=logits3.device)
            nll = -logprobs.gather(-1, targets[:, None]).sum().item()
            nll_sum += nll
            n_tokens += count
            prev_end = end
            if end == len(ids):
                break
    return float(torch.exp(torch.tensor(nll_sum / max(n_tokens, 1))))


def main():
    import os
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if len(t.strip()) > 200][:16]

    tok = QwenTokenizerAdapter("models/Qwen3-1.7B")
    results = {}
    for mode in ["fp16", "kv8", "kv4"]:
        cfg = EngineConfig(model_path="models/Qwen3-1.7B", device="cuda:0")
        eng = QslabEngine(cfg, kv_mode=mode, kv_plan_path="results/kv4_plan.json")
        # measure KV memory at a representative length (1024 tokens)
        eng.reset_cache()
        ids = tok.encode("x " * 1024)
        eng.prefill(ids)
        mem = sum(c.memory_bytes_valid() for c in eng.kv_caches) / 1e6
        ppl = ppl_engine(eng, tok, texts)
        results[mode] = {"ppl": round(ppl, 4), "kv_mem_mb@1024": round(mem, 3)}
        print(f"{mode:5s}: PPL {ppl:.4f} | KV mem @1024 ctx {mem:.3f} MB")
        del eng
        torch.cuda.empty_cache()

    import json
    from datetime import datetime
    out = {"model": "Qwen3-1.7B", "results": results,
           "timestamp": datetime.now().isoformat()}
    Path("results/m2_kv_ppl.json").write_text(json.dumps(out, indent=2))
    print("saved results/m2_kv_ppl.json")


if __name__ == "__main__":
    main()
