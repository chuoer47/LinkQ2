import math, time
import torch
import torch.nn.functional as F

def summarize_nll(rows):
    tokens = sum(row["tokens"] for row in rows)
    nll = math.fsum(row["nll"] for row in rows)
    if tokens <= 0 or not math.isfinite(nll):
        raise ValueError("non-finite or empty PPL result")
    return {"tokens": tokens, "ppl": math.exp(nll / tokens)}

@torch.inference_mode()
def quality(model, samples, device):
    started = time.perf_counter()
    rows = []
    for sample in samples:
        inputs = torch.tensor([sample["token_ids"]], device=device)
        output = model(input_ids=inputs, use_cache=False)
        nll = F.cross_entropy(output.logits[:, :-1].float().reshape(-1, output.logits.shape[-1]),
                              inputs[:, 1:].reshape(-1), reduction="sum").item()
        count = inputs.numel() - 1
        rows.append({"index": sample["index"], "start": sample["start"],
                     "tokens": count, "nll": nll, "ppl": math.exp(nll / count)})
        del inputs, output
        print(f"PPL sample {len(rows)}/{len(samples)} ppl={rows[-1]['ppl']:.6f} "
              f"elapsed={time.perf_counter()-started:.1f}s", flush=True)
    result = summarize_nll(rows)
    if result["tokens"] != sum(len(s["token_ids"]) - 1 for s in samples):
        raise AssertionError("PPL target accounting mismatch")
    return result
