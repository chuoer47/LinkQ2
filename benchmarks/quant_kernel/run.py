"""Run accuracy and speed cases for supported quantized linear kernels."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from benchmarks.quant_kernel.backends import marlin, nunchaku_awq, qslab_w4a16
from benchmarks.quant_kernel.cases import ALGORITHMS, BACKENDS, CASES, GROUP_SIZES, supports
from benchmarks.quant_kernel.metrics import measure
from benchmarks.quant_kernel.reference import compare, reference_linear
from qslab.compression.algorithms import QUANTIZERS

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results" / "quant_kernel"
ADAPTERS = {"qslab_w4a16": qslab_w4a16,
            "marlin": marlin, "nunchaku_awq": nunchaku_awq}


def query_gpus() -> list[dict[str, int | str]]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    gpus = []
    for row in result.stdout.splitlines():
        fields = [part.strip() for part in row.split(",", 4)]
        if len(fields) == 5:
            gpus.append({"index": int(fields[0]), "name": fields[1],
                         "memory_used_mib": int(fields[2]),
                         "memory_free_mib": int(fields[3]),
                         "utilization_percent": int(fields[4])})
    return gpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=(*BACKENDS, "all"), default="all")
    parser.add_argument("--case", choices=tuple(c["name"] for c in CASES) + ("all",),
                        default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    device = torch.device(args.device)
    gpu_state_before = query_gpus()
    selected_index = device.index or 0
    selected_before = next((g for g in gpu_state_before if g["index"] == selected_index), None)
    gpu_contended = bool(selected_before and (
        selected_before["memory_used_mib"] > 256
        or selected_before["utilization_percent"] > 10))
    selected_backends = BACKENDS if args.backend == "all" else (args.backend,)
    selected_cases = CASES if args.case == "all" else tuple(
        case for case in CASES if case["name"] == args.case)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    details = []
    for backend in selected_backends:
        algorithm = ALGORITHMS[backend]
        group_size = GROUP_SIZES[backend]
        for case in selected_cases:
            supported, reason = supports(backend, case)
            if not supported:
                details.append({"backend": backend, "case": case["name"],
                                "status": "skip", "reason": reason})
                continue
            m, n, k = case["m"], case["n"], case["k"]
            weight_fp = torch.randn(n, k, dtype=torch.float16) * 0.02
            x_cpu = torch.randn(m, k, dtype=torch.float16) * 0.1
            act_stat = torch.rand(k, dtype=torch.float16) + 0.1
            quantizer = QUANTIZERS[algorithm]
            quantized, act_scale = quantizer(
                weight_fp, group_size=group_size,
                activation_absmean=(act_stat if algorithm in ("awq", "nunchaku_awq")
                                    else None))
            x = x_cpu.to(device)
            expected = reference_linear(x, quantized, act_scale)
            adapter = ADAPTERS[backend]
            preparation_start = time.perf_counter()
            run = adapter.prepare(quantized, act_scale, device)
            torch.cuda.synchronize(device)
            preparation_ms = (time.perf_counter() - preparation_start) * 1000.0
            first_start = time.perf_counter()
            actual = run(x)
            torch.cuda.synchronize(device)
            first_call_ms = (time.perf_counter() - first_start) * 1000.0
            errors = compare(actual, expected)
            # Reference comparison is evaluated against the same packed logical W4.
            # Kernel-specific tolerances allow fp16 output rounding and reduction order.
            passed = (errors["relative_l2"] < 5e-3
                      and errors["cosine_similarity"] > 0.9999)
            performance = measure(run, x, args.warmup, args.repetitions)
            details.append({
                "backend": backend, "algorithm": algorithm,
                "case": case["name"], "m": m, "n": n, "k": k,
                "group_size": group_size, "status": "pass" if passed else "fail",
                "accuracy": errors, "performance": performance,
                "performance_may_be_contended": gpu_contended,
                "prepare_ms": preparation_ms, "first_call_ms": first_call_ms,
            })
            print(json.dumps(details[-1]), flush=True)
            del weight_fp, x_cpu, act_stat, quantized, act_scale
            del x, expected, actual, run
            torch.cuda.empty_cache()

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "gpu_state_before": gpu_state_before,
        "gpu_state_after": query_gpus(),
        "performance_may_be_contended": gpu_contended,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "seed": args.seed, "warmup": args.warmup,
        "repetitions": args.repetitions, "results": details,
    }
    path = args.output / "summary.json"
    path.write_text(json.dumps(summary, indent=2))
    failures = [r for r in details if r["status"] == "fail"]
    print(f"summary written to {path}; {len(failures)} accuracy failure(s)")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
