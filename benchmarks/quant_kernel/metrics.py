"""Synchronized operator latency and throughput measurements."""
from __future__ import annotations

import time

import torch


def measure(run, x: torch.Tensor, warmup: int, repetitions: int) -> dict[str, float | int]:
    for _ in range(warmup):
        output = run(x)
        torch.cuda.synchronize(x.device)
        del output
    samples = []
    for _ in range(repetitions):
        torch.cuda.synchronize(x.device)
        start = time.perf_counter()
        output = run(x)
        torch.cuda.synchronize(x.device)
        samples.append((time.perf_counter() - start) * 1000.0)
        del output
    ordered = sorted(samples)
    median = (ordered[(len(ordered) - 1) // 2] + ordered[len(ordered) // 2]) / 2
    p90 = ordered[min(len(ordered) - 1, int(0.90 * len(ordered)))]
    return {"repetitions": repetitions, "latency_median_ms": median,
            "latency_p90_ms": p90,
            "latency_method": "synchronized_host_wall",
            "rows_per_second": x.shape[0] * 1000.0 / median}


def wall_time_ms(fn) -> tuple[object, float]:
    start = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - start) * 1000.0
