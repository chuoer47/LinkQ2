"""Experiment identities, atomic records, GPU inspection, and subprocess metadata."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.real_quant.config import ROOT


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    temporary.replace(path)


def digest_json(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_files():
    folders = ("qslab", "adapters", "benchmarks/real_quant",
               "benchmarks/fake_quant_quality", "benchmarks/quant_kernel")
    return sorted(p for folder in folders for p in (ROOT / folder).rglob("*")
                  if p.is_file() and p.suffix in (".py", ".cu", ".cpp", ".h"))


def build_identity(model, data):
    model, data = Path(model), Path(data)
    model_files = sorted(p for p in model.iterdir() if p.is_file()
                         and (p.suffix == ".safetensors" or p.name in (
                             "config.json", "tokenizer.json", "tokenizer_config.json")))
    weights = {p.name: file_sha256(p) for p in model_files}
    datasets = {name: file_sha256(data / name) for name in
                ("calibration.pt", "wikitext_test.pt", "manifest.json")}
    sources = {str(p.relative_to(ROOT)): file_sha256(p) for p in source_files()}
    quant_sources = {name: sha for name, sha in sources.items() if name.startswith(
        ("qslab/compression/", "qslab/artifacts/", "qslab/conversion/",
         "adapters/", "qslab/reference/"))}
    identity = {"model": str(model.resolve()), "model_files": weights,
                "data": str(data.resolve()), "data_files": datasets,
                "source_files": sources, "dtype": "float16", "attention": "eager",
                "evaluation": "fixed independent token segments",
                "prefill_use_cache": False, "decode_use_cache": True}
    identity["environment"] = {name: importlib.metadata.version(name) for name in
        ("torch", "transformers", "nunchaku", "safetensors", "triton", "datasets", "tokenizers")}
    identity["environment"]["python"] = sys.version.split()[0]
    identity["artifact_identity"] = digest_json({"model": weights,
        "data": datasets, "sources": quant_sources, "dtype": "float16"})
    identity["identity"] = digest_json(identity)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True)
    identity["git_commit"] = revision.stdout.strip() if revision.returncode == 0 else None
    return identity


def query_gpus():
    result = subprocess.run(["nvidia-smi",
        "--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    rows = []
    for line in result.stdout.splitlines():
        index, uuid, name, used, free, util = [x.strip() for x in line.split(",")]
        rows.append({"index": int(index), "uuid": uuid, "name": name,
                     "memory_used_mib": int(used), "memory_free_mib": int(free),
                     "utilization_percent": int(util)})
    return rows


def selected_gpu(gpus, logical_index=0):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    selector = visible.split(",")[logical_index].strip() if visible else str(logical_index)
    return next((row for row in gpus if str(row["index"]) == selector
                 or row["uuid"].startswith(selector)), None)


def contended(gpu):
    return gpu is None or gpu["memory_used_mib"] > 256 or gpu["utilization_percent"] > 10


def artifact_path(root, identity, algorithm):
    return Path(root) / identity["artifact_identity"] / algorithm


def result_path(output, phase, algorithm, backend=None):
    label = algorithm + ("_" + backend if backend else "")
    return Path(output) / phase / (label + ".json")
