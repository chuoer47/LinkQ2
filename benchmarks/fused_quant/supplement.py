"""Reuse completed references; compare dispatch without changing upstream kernels."""
import argparse
import json
import os
import signal
import subprocess
import sys
import zipfile
from pathlib import Path

from benchmarks.real_quant.config import ROOT
from benchmarks.real_quant.common import write_json, file_sha256, utc_now
from benchmarks.fused_quant.run import summarize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--retry-failed", action="store_true")
    args = p.parse_args()
    output = args.output.resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("finish the original matrix before supplementing")
    record_path = output / "supplement_manifest.json"
    if record_path.exists():
        prior = json.loads(record_path.read_text())
        if not args.retry_failed or prior["status"] != "failed":
            raise ValueError("supplement already exists; only failed runs may be retried explicitly")
        archive = output / "attempts" / str(len(list((output / "attempts").glob("*"))) + 1)
        archive.mkdir(parents=True)
        record_path.rename(archive / record_path.name)
        (output / "source_direct.zip").rename(archive / "source_direct.zip")
        for job in prior["jobs"]:
            case_path = output / "cases" / (job["key"] + ".json")
            if case_path.exists() and json.loads(case_path.read_text())["status"] == "failed":
                case_path.rename(archive / case_path.name)
                log_path = output / "logs" / (job["key"] + ".log")
                log_path.rename(archive / log_path.name)
    specs = [("fp16", a, "fp32") for a in ("rtn", "rtn_clip", "awq", "nunchaku_awq")]
    specs += [("bf16", "rtn", "fp32"), ("fp16", "rtn", "fp16")]
    sources = sorted((ROOT / "benchmarks/fused_quant").glob("*.*"))
    record = {"status": "running", "started_utc": utc_now(),
        "source_hashes": {str(f.relative_to(ROOT)): file_sha256(f) for f in sources},
        "reused_reference_experiment": str(output), "matrix": specs,
        "dispatch": "direct upstream kernel.forward; same packing, routing and math",
        "jobs": []}
    with zipfile.ZipFile(output / "source_direct.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sources:
            z.write(f, str(f.relative_to(ROOT)))
    write_json(record_path, record)
    child = None
    def stop(signum, frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
        record.update(status="stopped", finished_utc=utc_now())
        write_json(record_path, record)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    for dtype, algorithm, accumulation in specs:
        reference = output / "cases" / f"{dtype}_{algorithm}_fake_fp32.json"
        assert json.loads(reference.read_text())["status"] == "completed"
        key = f"{dtype}_{algorithm}_gemlite_{accumulation}_direct"
        existing = output / "cases" / (key + ".json")
        if existing.exists() and json.loads(existing.read_text())["status"] == "completed":
            print(f"REUSE {key}", flush=True)
            continue
        command = [sys.executable, "-u", "-m", "benchmarks.fused_quant.worker", "--output", str(output),
            "--dtype", dtype, "--algorithm", algorithm, "--mode", "gemlite", "--dispatch", "direct",
            "--accumulation", accumulation]
        if algorithm == "rtn":
            command.append("--micro")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=manifest["gpu"], TOKENIZERS_PARALLELISM="false")
        with (output / "logs" / (key + ".log")).open("w") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            job = {"key": key, "command": command, "pid": child.pid, "started_utc": utc_now()}
            record["jobs"].append(job)
            write_json(record_path, record)
            print(f"START {key} pid={child.pid}", flush=True)
            code = child.wait()
            job.update(returncode=code, finished_utc=utc_now())
            write_json(record_path, record)
        summarize(output)
        print(f"END {key} returncode={code}", flush=True)
        if code:
            record.update(status="failed", failed_job=key, finished_utc=utc_now())
            write_json(record_path, record)
            raise SystemExit(code)
    record.update(status="completed", finished_utc=utc_now())
    write_json(record_path, record)


if __name__ == "__main__":
    main()
