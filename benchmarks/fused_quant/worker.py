"""One fresh CUDA process per precision, algorithm and execution path."""
import argparse
import importlib.metadata
import json
import statistics
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmarks.fused_quant.adapter import build_model, FusedLinear, inventory
from benchmarks.real_quant.common import (write_json, utc_now, query_gpus,
    selected_gpu, contended)
from benchmarks.real_quant.metrics import (cleanup, memory_snapshot, quality,
    probe, paired_probe, forward_timing, decode_timing)
from benchmarks.quant_kernel.reference import reference_linear, compare, dequantize
from qslab.artifacts.io import load_artifact


@torch.inference_mode()
def linear_checks(model, layers, artifact_dir, dtype):
    artifact = load_artifact(artifact_dir / "logical")
    checks = []
    generator = torch.Generator(device="cuda").manual_seed(2026)
    for row in layers[:7]:
        name = row["name"]
        w = artifact.weights[name + ".weight"]
        act = artifact.transforms.get("input_scale", {}).get(name + ".weight")
        for m in (1, 32, 128, 512):
            x = torch.randn((m, row["in_features"]), device="cuda", dtype=dtype,
                            generator=generator) * .1
            expected = reference_linear(x, w, act)
            module = model.get_submodule(name)
            actual = module(x)
            errors = compare(actual, expected)
            limit = .005 if dtype == torch.float16 else .01
            passed = errors["relative_l2"] < limit and errors["cosine_similarity"] > .9999
            equivalence = None
            if module.dispatch == "direct":
                module.dispatch = "module"
                try:
                    upstream_output = module(x)
                finally:
                    module.dispatch = "direct"
                equivalence = compare(actual, upstream_output)
                passed = passed and equivalence["relative_l2"] < .005
                del upstream_output
            checks.append({"layer": name, "m": m, **errors, "passed": passed,
                           "direct_vs_upstream_module": equivalence})
            del x, expected, actual
        print(f"LINEAR_CHECK {name} {sum(r['passed'] for r in checks)}/{len(checks)}", flush=True)
    return checks


@torch.inference_mode()
def measure(fn, repetitions=50):
    # First call includes compilation/autotuning and is never a latency sample.
    fn()
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    host = []
    for _ in range(20):
        start = time.perf_counter()
        y = fn()
        torch.cuda.synchronize()
        host.append((time.perf_counter() - start) * 1000)
        del y
    # A CUDA graph removes CPU launch gaps and measures steady GPU work.
    graph = torch.cuda.CUDAGraph()
    graph_calls = 32
    with torch.cuda.graph(graph):
        for _ in range(graph_calls):
            output = fn()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(repetitions)]
    for a, b in events:
        a.record()
        graph.replay()
        b.record()
    torch.cuda.synchronize()
    event_ms = [a.elapsed_time(b) / graph_calls for a, b in events]
    del graph, output
    return {"host_median_ms": statistics.median(host), "host_samples_ms": host,
            "cuda_graph_median_ms": statistics.median(event_ms), "cuda_graph_samples_ms": event_ms,
            "warmup": 10, "host_repetitions": 20, "graph_repetitions": repetitions,
            "calls_per_graph": graph_calls}


@torch.inference_mode()
def microbenchmark(model, layers, artifact_dir, model_path, dtype):
    from safetensors import safe_open
    artifact = load_artifact(artifact_dir / "logical")
    index = json.loads((Path(model_path) / "model.safetensors.index.json").read_text())["weight_map"]
    result = []
    seen = set()
    for row in layers[:7]:
        shape = row["out_features"], row["in_features"]
        if shape in seen:
            continue
        seen.add(shape)
        name = row["name"]
        key = name + ".weight"
        w = artifact.weights[key]
        act = artifact.transforms.get("input_scale", {}).get(key)
        fake = dequantize(w, torch.device("cuda"))
        if act is not None:
            fake = fake / act.to(device="cuda", dtype=torch.float32).reshape(1, -1)
        fake = fake.to(dtype)
        with safe_open(str(Path(model_path) / index[key]), framework="pt", device="cpu") as file:
            dense = file.get_tensor(key).to(device="cuda", dtype=dtype)
        for m in (1, 32, 128, 512):
            torch.manual_seed(2026 + m)
            x = torch.randn((m, shape[1]), device="cuda", dtype=dtype) * .1
            op = model.get_submodule(name)
            measurements = {}
            for path, fn in (("dense", lambda: F.linear(x, dense)),
                             ("fake", lambda: F.linear(x, fake)),
                             ("gemlite", lambda: op(x))):
                measurements[path] = measure(fn)
            result.append({"layer": name, "m": m, "n": shape[0], "k": shape[1],
                "measurements": measurements,
                "speedup_vs_dense_host": measurements["dense"]["host_median_ms"] / measurements["gemlite"]["host_median_ms"],
                "speedup_vs_dense_gpu": measurements["dense"]["cuda_graph_median_ms"] / measurements["gemlite"]["cuda_graph_median_ms"]})
            print(f"MICRO {name} M={m} GPU_speedup={result[-1]['speedup_vs_dense_gpu']:.3f}", flush=True)
        del fake, dense
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--algorithm", required=True)
    p.add_argument("--mode", choices=("dense", "fake", "gemlite"), required=True)
    p.add_argument("--dtype", choices=("fp16", "bf16"), required=True)
    p.add_argument("--accumulation", choices=("fp32", "fp16"), default="fp32")
    p.add_argument("--dispatch", choices=("module", "direct"), default="module")
    p.add_argument("--diagnostic-after-rejection", action="store_true")
    p.add_argument("--micro", action="store_true")
    args = p.parse_args()
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    key = f"{args.dtype}_{args.algorithm}_{args.mode}_{args.accumulation}"
    if args.dispatch == "direct":
        key += "_direct"
    if args.diagnostic_after_rejection:
        key += "_diagnostic"
    path = args.output / "cases" / (key + ".json")
    if path.exists():
        raise ValueError(f"case exists: {path}; do not overwrite completed evidence")
    manifest = json.loads((args.output / "manifest.json").read_text())
    artifact_dir = None if args.mode == "dense" else Path(manifest["artifact_root"]) / args.algorithm
    cache = args.output / (f"autotune_{args.dtype}_{args.accumulation}.json")
    row = {"key": key, "mode": args.mode, "algorithm": args.algorithm, "dtype": args.dtype,
        "accumulation": args.accumulation, "dispatch": args.dispatch,
        "started_utc": utc_now(), "status": "running",
        "sample_token_sha256": manifest["sample_token_sha256"], "operator_modified": False}
    row["eligible_for_speed_ranking"] = not args.diagnostic_after_rejection
    try:
        gpu = selected_gpu(query_gpus())
        if contended(gpu):
            raise RuntimeError(f"GPU occupied: {gpu}")
        row["gpu_before"] = gpu
        torch.cuda.set_device(0)
        torch.set_num_threads(4)
        torch.manual_seed(2026)
        torch.backends.cuda.matmul.allow_tf32 = False
        if args.mode == "gemlite":
            import gemlite
            # Reset bundled presets; autotune this GPU, persist configs per dtype/acc.
            gemlite.reset_config()
            if cache.exists():
                gemlite.load_config(str(cache))
        row["environment"] = {n: importlib.metadata.version(n) for n in
            ("torch", "triton", "transformers", "gemlite", "safetensors")}
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        print(f"LOAD {key}", flush=True)
        model, layers = build_model(manifest["model"], artifact_dir, args.mode, dtype,
                                    "cuda:0", args.accumulation, args.dispatch)
        row.update(load_seconds=time.perf_counter()-start, quantized_layers=len(layers),
                   load_peak_allocated_bytes=torch.cuda.max_memory_allocated())
        cleanup("cuda:0")
        row["weight_storage"] = inventory(model)
        row["resident_after_load"] = memory_snapshot("cuda:0")
        write_json(path, row)
        if args.mode == "gemlite":
            row["linear_checks"] = linear_checks(model, layers, artifact_dir, dtype)
            row["numerical_valid"] = all(c["passed"] for c in row["linear_checks"])
            gemlite.cache_config(str(cache))
            write_json(path, row)
            if not row["numerical_valid"] and not args.diagnostic_after_rejection:
                raise RuntimeError("fused operator numerical gate failed; inspect linear_checks")
        samples = json.loads((args.output / "samples.json").read_text())["samples"]
        ids = samples[0]["token_ids"]
        logits, counts = probe(model, ids[:32], "cuda:0", layers if args.mode == "gemlite" else [])
        paired_file = args.output / "paired" / f"{args.dtype}_{args.algorithm}_prefill.pt"
        if args.mode == "fake":
            paired_file.parent.mkdir(exist_ok=True)
            torch.save(logits, paired_file)
        elif args.mode == "gemlite":
            row["paired_probe"] = paired_probe(logits, torch.load(paired_file, weights_only=True))
            row["executed_layer_counts"] = counts
        del logits
        cleanup("cuda:0")
        row["resident_after_probe"] = memory_snapshot("cuda:0")
        torch.cuda.reset_peak_memory_stats()
        row["quality"] = quality(model, samples, "cuda:0")
        row["quality_memory"] = memory_snapshot("cuda:0")
        write_json(path, row)
        cleanup("cuda:0")
        row["prefill"] = forward_timing(model, ids, "cuda:0", (128, 512), 2, 5)
        write_json(path, row)
        row["decode"], decoded = decode_timing(model, ids, "cuda:0", 128, 32, 2, 5)
        paired_file = args.output / "paired" / f"{args.dtype}_{args.algorithm}_decode.pt"
        if args.mode == "fake":
            torch.save(decoded, paired_file)
        elif args.mode == "gemlite":
            row["paired_decode_probe"] = paired_probe(decoded, torch.load(paired_file, weights_only=True))
        del decoded
        write_json(path, row)
        if args.micro:
            row["microbenchmark"] = microbenchmark(model, layers, artifact_dir, manifest["model"], dtype)
        if args.mode == "gemlite":
            gemlite.cache_config(str(cache))
        del model
        cleanup("cuda:0")
        row.update(status="completed", finished_utc=utc_now(), gpu_after=selected_gpu(query_gpus()))
        write_json(path, row)
        print(f"COMPLETE {key}", flush=True)
    except Exception as exc:
        row.update(status="failed", error=repr(exc), traceback=traceback.format_exc(), finished_utc=utc_now())
        write_json(path, row)
        raise


if __name__ == "__main__":
    main()
