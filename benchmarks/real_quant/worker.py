"""One GPU process for one preparation, reference, real-quality, or timing job."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import time
import traceback
from pathlib import Path

import torch

from benchmarks.real_quant.common import (artifact_path, contended, file_sha256,
    query_gpus, result_path, selected_gpu, utc_now, write_json)
from benchmarks.real_quant.config import ALGORITHMS, BACKENDS, group_size
from benchmarks.real_quant.metrics import (check_linears, cleanup, decode_timing,
    forward_timing, memory_snapshot, paired_probe, probe, quality)
from benchmarks.real_quant.model import load_model, memory_inventory
from qslab.artifacts.io import load_artifact, save_artifact
from qslab.compression.config import CompressionConfig
from qslab.compression.pipeline import compress_checkpoint
from qslab.conversion.export_qslab_w4 import export_qslab_w4
from qslab.conversion.nunchaku_awq import export_nunchaku_awq


def prepare(args, identity, directory):
    directory.mkdir(parents=True, exist_ok=True)
    complete = directory / "complete.json"
    if complete.exists():
        row = json.loads(complete.read_text())
        if row["artifact_identity"] == identity["artifact_identity"]:
            if row.get("environment") != identity["environment"]:
                raise ValueError("cached artifact environment differs; use a new --artifact-root")
            for name, record in row["files"].items():
                p = directory / name
                if (not p.exists() or p.stat().st_size != record["bytes"]
                        or file_sha256(p) != record["sha256"]):
                    raise ValueError(f"incomplete cached artifact: {p}")
            print(f"reuse artifact {directory}", flush=True)
            return row
    start = time.perf_counter()
    calibration = str(Path(identity["data"]) / "calibration.pt")
    config = CompressionConfig(algorithm=args.algorithm, group_size=group_size(args.algorithm),
        device="cuda:0", artifact_device="cpu", calibration_path=calibration)
    print(f"compress {args.algorithm} using existing calibration", flush=True)
    artifact = compress_checkpoint(identity["model"], config)
    expected = artifact.model["num_hidden_layers"] * 7
    if len(artifact.weights) != expected:
        raise AssertionError(f"compression coverage {len(artifact.weights)} != {expected}")
    compress_seconds = time.perf_counter() - start
    save_artifact(artifact, directory / "logical")
    export_start = time.perf_counter()
    if args.algorithm == "nunchaku_awq":
        export_nunchaku_awq(artifact, directory / "nunchaku_awq")
    else:
        export_qslab_w4(artifact, directory / "qslab_w4")
    row = {"artifact_identity": identity["artifact_identity"], "environment": identity["environment"],
           "algorithm": args.algorithm,
           "group_size": group_size(args.algorithm), "quantized_layers": len(artifact.weights),
           "compress_seconds": compress_seconds,
           "export_seconds": time.perf_counter()-export_start,
           "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
           "files": {str(p.relative_to(directory)): {"bytes": p.stat().st_size,
                       "sha256": file_sha256(p)} for p in directory.rglob("*")
                      if p.is_file() and p.name != "complete.json"}}
    write_json(complete, row)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prepare", "reference", "real"), required=True)
    parser.add_argument("--algorithm", choices=("baseline", *ALGORITHMS), required=True)
    parser.add_argument("--backend", choices=BACKENDS)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--probe-tokens", type=int, default=32)
    parser.add_argument("--lengths", type=int, nargs="+", default=(128, 512))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--decode-context", type=int, default=128)
    parser.add_argument("--decode-steps", type=int, default=32)
    args = parser.parse_args()
    identity = json.loads(args.identity.read_text())
    path = result_path(args.output, args.phase, args.algorithm, args.backend)
    row = {"identity": identity["identity"], "phase": args.phase,
           "algorithm": args.algorithm, "backend": args.backend,
           "dtype": "float16", "attention": "eager", "use_cache": False,
           "started_utc": utc_now(), "status": "running", "full_evaluation": False,
           "protocol": "small_paired_v2"}
    try:
        before = query_gpus()
        gpu = selected_gpu(before)
        row.update(gpu_state_before=before, selected_gpu=gpu,
                   performance_may_be_contended=contended(gpu))
        if contended(gpu):
            raise RuntimeError(f"selected GPU is occupied before this job: {gpu}")
        torch.cuda.set_device(0)
        torch.set_num_threads(4)
        torch.manual_seed(2026)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        row["environment"] = {name: importlib.metadata.version(name) for name in
            ("torch", "transformers", "nunchaku", "safetensors", "triton")}
        row["environment"].update(torch_cuda=torch.version.cuda,
                                   torch_build=torch.__version__)
        directory = None if args.algorithm == "baseline" else artifact_path(
            args.artifact_root, identity, args.algorithm)
        if args.phase == "prepare":
            row["artifact"] = prepare(args, identity, directory)
        else:
            manifest = json.loads((args.output / "samples.json").read_text())
            samples = manifest["samples"]
            ids = samples[0]["token_ids"]
            row["sample_token_sha256"] = manifest["token_sha256"]
            row["timing_sample_index"] = samples[0]["index"]
            row["hardware_after_cuda_init"] = selected_gpu(query_gpus())
            start = time.perf_counter()
            model, count, layers = load_model(identity["model"], args.phase,
                args.backend, directory, "cuda:0")
            row.update(load_seconds=time.perf_counter()-start, quantized_layers=count,
                       layers=layers, load_peak_allocated_bytes=torch.cuda.max_memory_allocated())
            row["operator_path"] = {"backend": args.backend,
                "maximum_rows_per_call": {"nunchaku_awq": 8}.get(args.backend),
                "dequantization": "inside CUDA operator" if args.backend else None}
            gc.collect()
            torch.cuda.empty_cache()
            row["memory"] = memory_inventory(model)
            row["resident_after_load"] = memory_snapshot("cuda:0")
            row["hardware_after_load"] = selected_gpu(query_gpus())
            probe_ids = ids[:args.probe_tokens]
            row["probe_tokens"] = len(probe_ids)
            if args.phase == "real":
                artifact = load_artifact(directory / "logical")
                check_rows = tuple(sorted({1, args.probe_tokens, args.decode_context, *args.lengths}))
                row["linear_checks"] = check_linears(model, artifact, layers, "cuda:0", rows=check_rows)
                del artifact
            logits, counts = probe(model, probe_ids, "cuda:0", layers)
            if args.phase == "reference":
                probe_path = args.output / "reference" / (args.algorithm + "_probe.pt")
                probe_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"input_ids": probe_ids, "logits": logits}, probe_path)
            else:
                expected = torch.load(args.output / "reference" / (args.algorithm + "_probe.pt"),
                                      map_location="cpu", weights_only=True)
                if expected["input_ids"] != probe_ids:
                    raise ValueError("paired probe inputs differ")
                row["paired_probe"] = paired_probe(logits, expected["logits"])
                row["executed_layer_counts"] = counts
                del expected
            del logits
            cleanup("cuda:0")
            row["resident_after_probe"] = memory_snapshot("cuda:0")
            row["hardware_after_probe"] = selected_gpu(query_gpus())
            torch.cuda.reset_peak_memory_stats()
            row["quality"] = quality(model, samples, "cuda:0")
            row["evaluation_memory"] = memory_snapshot("cuda:0")
            write_json(path, row)
            cleanup("cuda:0")
            row["prefill"] = forward_timing(model, ids, "cuda:0", args.lengths,
                                           args.warmup, args.repetitions)
            write_json(path, row)
            row["decode"], decoded_logits = decode_timing(model, ids, "cuda:0",
                args.decode_context, args.decode_steps, args.warmup, args.repetitions)
            decode_path = args.output / "reference" / (args.algorithm + "_decode_probe.pt")
            decode_ids = ids[:args.decode_context + args.decode_steps]
            if args.phase == "reference":
                torch.save({"input_ids": decode_ids, "logits": decoded_logits}, decode_path)
            else:
                expected = torch.load(decode_path, map_location="cpu", weights_only=True)
                if expected["input_ids"] != decode_ids:
                    raise ValueError("paired decode inputs differ")
                row["paired_decode_probe"] = paired_probe(decoded_logits, expected["logits"])
                del expected
            del decoded_logits
            del model
        row.update(status="completed", finished_utc=utc_now(), gpu_state_after=query_gpus())
        write_json(path, row)
        print(f"RESULT {path}", flush=True)
    except Exception as error:
        row.update(status="failed", finished_utc=utc_now(), error=repr(error),
                   traceback=traceback.format_exc())
        write_json(path, row)
        raise


if __name__ == "__main__":
    main()
