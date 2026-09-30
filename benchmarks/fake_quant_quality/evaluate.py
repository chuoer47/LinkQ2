"""Compute token-weighted causal language-model PPL."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.inference_mode()
def evaluate_ppl(model: torch.nn.Module, token_ids: list[int], device: str,
                 max_length: int, stride: int) -> dict[str, float | int]:
    stream = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    total_nll = 0.0
    total_tokens = 0
    previous_end = 0
    for end in range(min(max_length, stream.shape[1]), stream.shape[1] + stride, stride):
        end = min(end, stream.shape[1])
        begin = max(0, end - max_length)
        inputs = stream[:, begin:end]
        outputs = model(input_ids=inputs, use_cache=False)
        labels = inputs[:, 1:]
        logits = outputs.logits[:, :-1, :].float()
        target_start = max(previous_end, begin + 1)
        local_start = target_start - begin - 1
        if local_start < labels.shape[1]:
            losses = F.cross_entropy(
                logits[:, local_start:].reshape(-1, logits.shape[-1]),
                labels[:, local_start:].reshape(-1), reduction="sum")
            count = labels[:, local_start:].numel()
            total_nll += float(losses.item())
            total_tokens += count
        previous_end = end
        if end == stream.shape[1]:
            break
    if total_tokens == 0:
        raise RuntimeError("evaluation produced no prediction targets")
    mean_nll = total_nll / total_tokens
    return {"nll": total_nll, "tokens": total_tokens,
            "mean_nll": mean_nll, "ppl": math.exp(mean_nll)}
