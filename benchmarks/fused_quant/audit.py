"""Verify saved measurements, source snapshots and untouched upstream packages."""
import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    output = args.output.resolve()
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((output / "manifest.json").read_text())
    audit = json.loads((output / "audit_before.json").read_text())
    for relative, expected in audit["original_operator_files"].items():
        assert sha(root / relative) == expected, relative
    for relative, expected in audit["gemlite_python_files"].items():
        assert sha(root / ".venvs/gemlite-0.4.8/lib/python3.11/site-packages" / relative) == expected, relative
    for metadata, archive in (("manifest.json", "source.zip"),
                              ("supplement_manifest.json", "source_direct.zip"),
                              ("diagnostic_manifest.json", "source_diagnostic.zip")):
        records = json.loads((output / metadata).read_text())["source_hashes"]
        with zipfile.ZipFile(output / archive) as source:
            for relative, expected in records.items():
                assert hashlib.sha256(source.read(relative)).hexdigest() == expected, (archive, relative)
    cases = [json.loads(file.read_text()) for file in (output / "cases").glob("*.json")]
    completed = [a for a in cases if a["status"] == "completed"]
    valid = [a for a in completed if all(c["passed"] for c in a.get("linear_checks", []))]
    assert len(completed) == 18 and len(valid) == 17, (len(completed), len(valid))
    assert sum(a["mode"] == "gemlite" for a in valid) == 10
    for case in completed:
        assert case["quality"]["tokens"] == 4088 and len(case["quality"]["samples"]) == 8
        assert case["sample_token_sha256"] == manifest["sample_token_sha256"]
        decode = case["decode"]
        assert decode["kv_cache_initial"]["sequence_length"] == 128
        assert decode["kv_cache_final"]["sequence_length"] == 160
        dtype = "torch.float16" if case["dtype"] == "fp16" else "torch.bfloat16"
        assert decode["kv_cache_final"]["dtypes"] == [dtype]
        if case["mode"] == "gemlite":
            assert len(case["executed_layer_counts"]) == 252
            assert all(v == 1 for v in case["executed_layer_counts"].values())
    original = json.loads(subprocess.check_output([
        "/home/amax01/miniconda3/envs/qslab/bin/python", "-c",
        "import importlib.metadata,json;print(json.dumps({n:importlib.metadata.version(n) for n in ('torch','triton','transformers','safetensors')}))"], text=True))
    assert original == manifest["original_conda_environment"] and original["triton"] == "3.1.0"
    audit.update(upstream_gemlite_unchanged_after=True, original_operators_unchanged_after=True,
        original_conda_unchanged=True, original_conda_environment=original,
        completed_measurements=18, completed_valid=17, real_valid=10,
        numerically_rejected_configurations=1, diagnostic_measurements=1,
        source_snapshots_verified=True)
    old = json.loads((Path(manifest["previous"]) / "identity.json").read_text())
    audit["model_hashes_verified"] = {}
    for name, expected in old["model_files"].items():
        actual = sha(Path(old["model"]) / name)
        assert actual == expected, name
        audit["model_hashes_verified"][name] = actual
        print("MODEL_HASH_VERIFIED", name, flush=True)
    for name, expected in old["data_files"].items():
        assert sha(Path(old["data"]) / name) == expected, name
    jobs = json.loads((output / "jobs.json").read_text())
    jobs += json.loads((output / "supplement_manifest.json").read_text())["jobs"]
    jobs += [json.loads((output / "diagnostic_manifest.json").read_text())]
    alive = []
    for job in jobs:
        cmdline = Path("/proc") / str(job["pid"]) / "cmdline"
        if cmdline.exists() and b"benchmarks.fused_quant.worker" in cmdline.read_bytes():
            alive.append(job["pid"])
    assert not alive, alive
    audit["owned_gpu_workers_alive"] = alive
    audit["gpu_final"] = subprocess.check_output(["nvidia-smi",
        "--query-gpu=index,name,memory.used,utilization.gpu", "--format=csv,noheader"], text=True).splitlines()
    audit["paired_probe_files_kept_on_server"] = {str(file.relative_to(output)):
        {"bytes": file.stat().st_size, "sha256": sha(file)} for file in (output / "paired").glob("*.pt")}
    (output / "audit_after.json").write_text(json.dumps(audit, indent=2))
    print("AUDIT_OK", audit["gpu_final"], flush=True)


if __name__ == "__main__":
    main()
