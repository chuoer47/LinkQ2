"""Small paired PPL, unmodified-operator checks, and HF prefill/decode timing."""
from __future__ import annotations

import gc
import math
import statistics
import time

import torch
import torch.nn.functional as F



@torch.inference_mode()
def probe(model, ids, device, layers):
    counts = {row["name"]: 0 for row in layers}
    handles = []
    for name in counts:
        def hook(_module, _args, _output, key=name):
            counts[key] += 1
        handles.append(model.get_submodule(name).register_forward_hook(hook))
    try:
        inputs = torch.tensor([ids], device=device)
        output = model(input_ids=inputs, use_cache=False)
        logits = output.logits.float().cpu()
        del output, inputs
    finally:
        for handle in handles:
            handle.remove()
    if counts and any(count != 1 for count in counts.values()):
        raise AssertionError(f"packed layer execution coverage failed: {counts}")
    return logits, counts


def paired_probe(actual, expected):
    a, b = actual.double().flatten(), expected.double().flatten()
    delta = a - b
    denominator = b.norm().clamp_min(1e-12)
    cosine = (a.dot(b) / (a.norm().clamp_min(1e-12) * denominator)).clamp(-1, 1)
    metrics = {"max_abs": float(delta.abs().max()),
               "rms": float(delta.square().mean().sqrt()),
               "relative_l2": float(delta.norm() / denominator),
               "cosine_similarity": float(cosine)}
    metrics["top1_agreement"] = float(
        (actual.argmax(-1) == expected.argmax(-1)).float().mean().item())
    return metrics


def summarize_nll(rows):
    tokens = sum(row["tokens"] for row in rows)
    nll = math.fsum(row["nll"] for row in rows)
    if tokens <= 0 or not math.isfinite(nll):
        raise ValueError("non-finite or empty PPL result")
    return {"nll": nll, "tokens": tokens, "mean_nll": nll / tokens,
            "ppl": math.exp(nll / tokens), "samples": rows}


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
    result["evaluation_seconds"] = time.perf_counter() - started
    if result["tokens"] != sum(len(s["token_ids"]) - 1 for s in samples):
        raise AssertionError("PPL target accounting mismatch")
    return result


def memory_snapshot(device):
    return {"allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def cleanup(device):
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()


def latency_summary(samples):
    ordered = sorted(samples)
    return {"latency_median_ms": statistics.median(samples),
            "latency_p90_ms": ordered[math.ceil(0.9 * len(ordered)) - 1],
            "latency_samples_ms": samples}


@torch.inference_mode()
def forward_timing(model, ids, device, lengths, warmup, repetitions):
    rows = []
    for length in lengths:
        if length > len(ids):
            raise ValueError("prefill input exceeds the saved sample")
        inputs = torch.tensor([ids[:length]], device=device)
        cleanup(device)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        output = model(input_ids=inputs, use_cache=False)
        torch.cuda.synchronize(device)
        first_ms = (time.perf_counter() - start) * 1000
        del output
        for _ in range(warmup):
            output = model(input_ids=inputs, use_cache=False)
            torch.cuda.synchronize(device)
            del output
        cleanup(device)
        torch.cuda.reset_peak_memory_stats(device)
        samples = []
        for i in range(repetitions):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            output = model(input_ids=inputs, use_cache=False)
            torch.cuda.synchronize(device)
            samples.append((time.perf_counter() - start) * 1000)
            del output
            print(f"PREFILL S={length} repeat={i+1}/{repetitions} ms={samples[-1]:.3f}", flush=True)
        stats = latency_summary(samples)
        rows.append({"batch_size": 1, "sequence_length": length, "use_cache": False,
                     "includes_lm_head": True, "first_measured_forward_ms": first_ms,
                     **stats, "input_tokens_per_second": length * 1000 / stats["latency_median_ms"],
                     "warmup": warmup, "repetitions": repetitions,
                     "latency_method": "synchronized_host_wall", "memory": memory_snapshot(device)})
        del inputs
    cleanup(device)
    return rows


def cache_inventory(cache):
    if hasattr(cache, "layers"):
        tensors = [t for layer in cache.layers for t in
                   (getattr(layer, "keys", None), getattr(layer, "values", None))]
    elif hasattr(cache, "key_cache"):
        tensors = list(cache.key_cache) + list(cache.value_cache)
    else:
        tensors = [t for layer in cache for t in layer[:2]]
    seen, total, dtypes = set(), 0, set()
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            continue
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        if key not in seen:
            seen.add(key)
            total += storage.nbytes()
            dtypes.add(str(tensor.dtype))
    return {"storage_bytes": total, "dtypes": sorted(dtypes),
            "sequence_length": int(cache.get_seq_length()) if hasattr(cache, "get_seq_length") else None}


@torch.inference_mode()
def decode_cycle(model, inputs, context, steps, device, capture_logits=False):
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    output = model(input_ids=inputs[:, :context], use_cache=True)
    torch.cuda.synchronize(device)
    context_ms = (time.perf_counter() - start) * 1000
    cache = output.past_key_values
    del output
    initial_cache = cache_inventory(cache)
    timings, logits = [], []
    for step in range(steps):
        token = inputs[:, context + step:context + step + 1]
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        output = model(input_ids=token, past_key_values=cache, use_cache=True)
        torch.cuda.synchronize(device)
        timings.append((time.perf_counter() - start) * 1000)
        cache = output.past_key_values
        if capture_logits:
            logits.append(output.logits.float().cpu())
        del output
    final_cache = cache_inventory(cache)
    if final_cache["sequence_length"] != context + steps:
        raise AssertionError(f"decode cache did not grow: {final_cache}")
    del cache
    return {"step_ms": timings, "context_prefill_ms": context_ms,
            "kv_cache_initial": initial_cache, "kv_cache_final": final_cache,
            "logits": torch.cat(logits, dim=1) if capture_logits else None}


@torch.inference_mode()
def decode_timing(model, ids, device, context, steps, warmup, repetitions):
    if context + steps > len(ids):
        raise ValueError("decode tokens exceed the saved sample")
    inputs = torch.tensor([ids[:context + steps]], device=device)
    cleanup(device)
    first = decode_cycle(model, inputs, context, steps, device)
    for _ in range(warmup):
        decode_cycle(model, inputs, context, steps, device)
    cleanup(device)
    torch.cuda.reset_peak_memory_stats(device)
    cycles = []
    for i in range(repetitions):
        cycle = decode_cycle(model, inputs, context, steps, device)
        cycles.append(cycle)
        print(f"DECODE repeat={i+1}/{repetitions} mean_ms/token={statistics.mean(cycle['step_ms']):.3f}", flush=True)
    means = [statistics.mean(c["step_ms"]) for c in cycles]
    stats = latency_summary(means)
    pooled = latency_summary([ms for c in cycles for ms in c["step_ms"]])
    memory = memory_snapshot(device)
    captured = decode_cycle(model, inputs, context, steps, device, capture_logits=True)
    result = {"batch_size": 1, "context_tokens": context, "decode_steps": steps,
              "use_cache": True, "cache_class": "HF dynamic cache", "includes_lm_head": True,
              "teacher_forced": True, "warmup": warmup, "repetitions": repetitions,
              "latency_method": "synchronized_host_wall_per_step",
              **stats, "tokens_per_second": 1000 / stats["latency_median_ms"],
              "pooled_step_median_ms": pooled["latency_median_ms"],
              "pooled_step_p90_ms": pooled["latency_p90_ms"],
              "context_prefill_samples_ms": [c["context_prefill_ms"] for c in cycles],
              "first_cycle_mean_ms": statistics.mean(first["step_ms"]),
              "step_samples_ms": [c["step_ms"] for c in cycles],
              "kv_cache_initial": cycles[-1]["kv_cache_initial"],
              "kv_cache_final": cycles[-1]["kv_cache_final"], "memory": memory}
    del inputs
    cleanup(device)
    return result, captured["logits"]
