"""Run the small paired real-quant benchmark without changing operators."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from benchmarks.real_quant.common import (build_identity, digest_json, query_gpus,
    result_path, source_files, utc_now, write_json)
from benchmarks.real_quant.config import (ALGORITHMS, ARTIFACTS, BACKENDS, DATA,
    DECODE_CONTEXT, DECODE_STEPS, MODEL, PREFILL_LENGTHS, ROOT, SAMPLE_COUNT,
    SAMPLE_LENGTH, SAMPLE_SEED, matrix)
from benchmarks.real_quant.samples import select_samples

_STOP = threading.Event()
_ACTIVE = set()
_LOCK = threading.Lock()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=MODEL)
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--artifact-root", type=Path, default=ARTIFACTS)
    p.add_argument("--output", type=Path)
    p.add_argument("--devices", nargs="+", default=("1",), help="physical GPU indices")
    p.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=ALGORITHMS)
    p.add_argument("--backends", nargs="+", choices=BACKENDS, default=BACKENDS)
    p.add_argument("--sample-count", type=int, default=SAMPLE_COUNT)
    p.add_argument("--sample-length", type=int, default=SAMPLE_LENGTH)
    p.add_argument("--sample-seed", type=int, default=SAMPLE_SEED)
    p.add_argument("--probe-tokens", type=int, default=32)
    p.add_argument("--lengths", type=int, nargs="+", default=PREFILL_LENGTHS)
    p.add_argument("--decode-context", type=int, default=DECODE_CONTEXT)
    p.add_argument("--decode-steps", type=int, default=DECODE_STEPS)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repetitions", type=int, default=5)
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-ppl-drift", type=float, default=0.01,
                   help="relative real/fake PPL warning threshold; does not suppress results")
    return p


def stop_children(_sig, _frame):
    _STOP.set()
    with _LOCK:
        for process in tuple(_ACTIVE):
            if process.poll() is None:
                process.terminate()
    raise KeyboardInterrupt("user requested termination")


def execute(args, identity, job, gpu):
    phase, algorithm, backend = job
    label = phase + "_" + algorithm + ("_" + backend if backend else "")
    expected = digest_json({"identity": identity["identity"], "phase": phase,
        "algorithm": algorithm, "backend": backend, "protocol": identity["protocol"]})
    path = result_path(args.output, phase, algorithm, backend)
    record = args.output / "jobs" / (label + ".json")
    if path.exists() and record.exists() and not args.force:
        previous, previous_job = json.loads(path.read_text()), json.loads(record.read_text())
        if previous.get("status") == "completed" and previous_job.get("job_identity") == expected:
            print(f"REUSE {label}", flush=True)
            return previous_job
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
               TORCH_CUDA_ARCH_LIST="8.9", MAX_JOBS="4")
    prefix = Path(sys.prefix)
    env.setdefault("CUDA_HOME", str(prefix))
    for key, binary in (("CC", "x86_64-conda-linux-gnu-gcc"),
                        ("CXX", "x86_64-conda-linux-gnu-g++")):
        if (prefix / "bin" / binary).exists():
            env[key] = str(prefix / "bin" / binary)
    command = [sys.executable, "-u", "-m", "benchmarks.real_quant.worker", "--phase", phase,
        "--algorithm", algorithm, "--identity", str(args.output / "identity.json"),
        "--output", str(args.output), "--artifact-root", str(args.artifact_root),
        "--probe-tokens", str(args.probe_tokens), "--warmup", str(args.warmup),
        "--repetitions", str(args.repetitions), "--decode-context", str(args.decode_context),
        "--decode-steps", str(args.decode_steps), "--lengths", *map(str, args.lengths)]
    if backend:
        command += ["--backend", backend]
    log = args.output / "logs" / (label + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    row = {"job_identity": expected, "phase": phase, "algorithm": algorithm,
           "backend": backend, "physical_gpu": gpu, "command": command,
           "log": str(log), "status": "running", "started_utc": utc_now()}
    print(f"START {label} gpu={gpu}", flush=True)
    started = time.perf_counter()
    with log.open("w") as file:
        with _LOCK:
            if _STOP.is_set():
                raise KeyboardInterrupt("experiment stopped")
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=file,
                                       stderr=subprocess.STDOUT)
            _ACTIVE.add(process)
        row["pid"] = process.pid
        write_json(record, row)
        rc = process.wait()
        with _LOCK:
            _ACTIVE.discard(process)
    row.update(status="completed" if rc == 0 else "failed", exit_code=rc,
               finished_utc=utc_now(), wall_seconds=time.perf_counter()-started)
    write_json(record, row)
    print(f"END {label} status={row['status']} seconds={row['wall_seconds']:.1f}", flush=True)
    return row


def run_phase(args, identity, jobs, devices):
    pending = queue.Queue()
    for job in jobs:
        pending.put(job)

    def consume(gpu):
        rows = []
        while not _STOP.is_set():
            try:
                job = pending.get_nowait()
            except queue.Empty:
                break
            row = execute(args, identity, job, gpu)
            rows.append(row)
            if job[0] == "reference" and row["status"] != "completed":
                break
        return rows

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(consume, gpu) for gpu in devices]
        return [row for future in futures for row in future.result()]


def read_result(output, phase, algorithm, backend=None):
    path = result_path(output, phase, algorithm, backend)
    return json.loads(path.read_text()) if path.exists() else {"status": "missing"}


def aggregate(args, identity, pairs):
    baseline = read_result(args.output, "reference", "baseline")
    fake_rows = {a: read_result(args.output, "reference", a) for a, _ in pairs}
    results = []
    for algorithm, backend in pairs:
        fake = fake_rows[algorithm]
        real = read_result(args.output, "real", algorithm, backend)
        row = {"algorithm": algorithm, "backend": backend,
               "group_size": 64 if algorithm == "nunchaku_awq" else 128,
               "execution_status": real["status"], "raw_result": real}
        if real["status"] != "completed" or fake["status"] != "completed" or baseline["status"] != "completed":
            row.update(validity="execution_failed", error=real.get("error", "paired reference unavailable"))
            results.append(row)
            continue
        drift = real["quality"]["ppl"] / fake["quality"]["ppl"] - 1
        checks = real["linear_checks"]
        failures = [c for c in checks if c["status"] != "pass"]
        probe_ok = real["paired_probe"]["relative_l2"] < 0.02
        decode_ok = real["paired_decode_probe"]["relative_l2"] < 0.02
        ppl_ok = abs(drift) <= args.max_ppl_drift
        valid = not failures and probe_ok and decode_ok and ppl_ok
        row.update(validity="valid" if valid else "numerical_warning", fake_ppl=fake["quality"]["ppl"],
            real_ppl=real["quality"]["ppl"], ppl_absolute_delta=real["quality"]["ppl"]-fake["quality"]["ppl"],
            ppl_relative_drift=drift, ppl_drift_warning=not ppl_ok,
            linear_checks_passed=len(checks)-len(failures), linear_checks_total=len(checks),
            failed_linear_checks=failures, paired_probe=real["paired_probe"],
            paired_decode_probe=real["paired_decode_probe"],
            resident_weight_bytes=real["memory"]["resident_weight_bytes"],
            resident_allocated_bytes=real["resident_after_probe"]["allocated_bytes"],
            resident_saving_vs_fake=1-real["resident_after_probe"]["allocated_bytes"]/fake["resident_after_probe"]["allocated_bytes"],
            resident_saving_vs_fp16=1-real["resident_after_probe"]["allocated_bytes"]/baseline["resident_after_probe"]["allocated_bytes"],
            prefill=[])
        for timing in real["prefill"]:
            length = timing["sequence_length"]
            ft = next(t for t in fake["prefill"] if t["sequence_length"] == length)
            bt = next(t for t in baseline["prefill"] if t["sequence_length"] == length)
            mode_ok = probe_ok and ppl_ok and all(c["status"] == "pass" for c in checks if c["m"] == length)
            row["prefill"].append({**timing, "validity": "valid" if mode_ok else "numerical_warning",
                "observed_speedup_vs_fake": ft["latency_median_ms"]/timing["latency_median_ms"],
                "observed_speedup_vs_fp16": bt["latency_median_ms"]/timing["latency_median_ms"],
                "peak_saving_vs_fake": 1-timing["memory"]["peak_allocated_bytes"]/ft["memory"]["peak_allocated_bytes"],
                "peak_saving_vs_fp16": 1-timing["memory"]["peak_allocated_bytes"]/bt["memory"]["peak_allocated_bytes"]})
        dt, fd, bd = real["decode"], fake["decode"], baseline["decode"]
        decode_valid = decode_ok and all(c["status"] == "pass" for c in checks if c["m"] in (1, args.decode_context))
        row["decode"] = {**dt, "validity": "valid" if decode_valid else "numerical_warning",
            "observed_speedup_vs_fake": fd["latency_median_ms"]/dt["latency_median_ms"],
            "observed_speedup_vs_fp16": bd["latency_median_ms"]/dt["latency_median_ms"],
            "peak_saving_vs_fake": 1-dt["memory"]["peak_allocated_bytes"]/fd["memory"]["peak_allocated_bytes"],
            "peak_saving_vs_fp16": 1-dt["memory"]["peak_allocated_bytes"]/bd["memory"]["peak_allocated_bytes"]}
        results.append(row)
    completed = sum(r["execution_status"] == "completed" for r in results)
    status = "completed" if completed == len(pairs) and all(f["status"] == "completed" for f in (baseline, *fake_rows.values())) else "completed_with_errors"
    summary = {"created_utc": utc_now(), "status": status, "identity": identity["identity"],
        "model": identity["model"], "protocol": identity["protocol"],
        "full_evaluation": False, "scope": "HF eager prefill + teacher-forced decode; packed W4; original operators; no inference engine",
        "baseline": baseline, "fake_references": fake_rows,
        "max_relative_ppl_drift": args.max_ppl_drift, "results": results}
    write_json(args.output / "summary.json", summary)
    write_report(args.output, summary)
    return summary


def write_report(output, summary):
    protocol, base = summary["protocol"], summary["baseline"]
    lines = ["# Real-quant 小样本实验结果", "",
        f"模型：`{summary['model']}`；状态：`{summary['status']}`。",
        "HF eager 执行完整模型，使用原始算子；prefill 无 KV cache，decode 使用普通 FP16 KV cache。",
        f"PPL：固定 {protocol['sample_count']}×{protocol['sample_length']} token，"
        f"有效预测目标 {protocol['scored_tokens']} 个；样本 SHA256：`{protocol['sample_token_sha256']}`。",
        f"预热 {protocol['warmup']} 次、正式 {protocol['repetitions']} 次；各配置在同一张 GPU 顺序测量。",
        "", "## PPL 与正确性", "",
        "| 配置 | Fake PPL | Real PPL | 绝对差 | 相对差 | 算子检查 | 判定 |",
        "|---|---:|---:|---:|---:|---:|---|"]
    if base.get("quality"):
        lines.insert(5, f"FP16 小样本 PPL：{base['quality']['ppl']:.6f}。")
    good = [r for r in summary["results"] if "real_ppl" in r]
    for r in good:
        lines.append(f"| {r['algorithm']}/{r['backend']} | {r['fake_ppl']:.6f} | {r['real_ppl']:.6f} | "
            f"{r['ppl_absolute_delta']:+.6f} | {100*r['ppl_relative_drift']:+.4f}% | "
            f"{r['linear_checks_passed']}/{r['linear_checks_total']} | {r['validity']} |")
    lines += ["", "## Prefill 性能", "",
        "延迟包括 attention、所有投影和 LM head，按同步后的主机墙钟计时。加速比大于 1 表示更快。",
        "", "| 配置 | 输入长度 | 中位数 ms | P90 ms | token/s | 相对 Fake | 相对 FP16 | 判定 |",
        "|---|---:|---:|---:|---:|---:|---:|---|"]
    for label, result in [("FP16", base)] + [("fake/"+a,f) for a,f in summary["fake_references"].items()]:
        for t in result.get("prefill", []):
            lines.append(f"| {label} | {t['sequence_length']} | {t['latency_median_ms']:.3f} | {t['latency_p90_ms']:.3f} | {t['input_tokens_per_second']:.1f} | — | — | reference |")
    for r in good:
        for t in r["prefill"]:
            lines.append(f"| {r['algorithm']}/{r['backend']} | {t['sequence_length']} | {t['latency_median_ms']:.3f} | "
                f"{t['latency_p90_ms']:.3f} | {t['input_tokens_per_second']:.1f} | {t['observed_speedup_vs_fake']:.3f}× | "
                f"{t['observed_speedup_vs_fp16']:.3f}× | {t['validity']} |")
    lines += ["", "## Decode 性能", "",
        f"固定 {protocol['decode_context']} token 上下文，随后固定输入 {protocol['decode_steps']} 个 token。"
        "每轮重新建立 cache；上下文 prefill 不计入 decode 延迟。",
        "每轮取 32 步平均 ms/token，再对 5 轮取中位数和 P90；完整逐步数据保存在 JSON。",
        "", "| 配置 | 中位数 ms/token | P90 | token/s | 相对 Fake | 相对 FP16 | 判定 |",
        "|---|---:|---:|---:|---:|---:|---|"]
    for label, result in [("FP16",base)] + [("fake/"+a,f) for a,f in summary["fake_references"].items()]:
        if "decode" in result:
            t = result["decode"]
            lines.append(f"| {label} | {t['latency_median_ms']:.3f} | {t['latency_p90_ms']:.3f} | {t['tokens_per_second']:.1f} | — | — | reference |")
    for r in good:
        t = r["decode"]
        lines.append(f"| {r['algorithm']}/{r['backend']} | {t['latency_median_ms']:.3f} | {t['latency_p90_ms']:.3f} | "
            f"{t['tokens_per_second']:.1f} | {t['observed_speedup_vs_fake']:.3f}× | {t['observed_speedup_vs_fp16']:.3f}× | {t['validity']} |")
    lines += ["", "## 显存", "",
        "GiB=2^30 字节。常驻值为清理临时张量后的 PyTorch allocated；模型存储包含 packed 权重、scale/zero、未量化层及 workspace。",
        "运行峰值含模型、激活及输出，decode 峰值还包含上下文 prefill 与 KV cache。reserved 和硬件快照见原始 JSON。",
        "", "| 配置 | 模型存储 GiB | 常驻 GiB | 相对 Fake 节省 | Prefill 512 峰值 GiB | Decode 峰值 GiB | KV cache MiB |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for label, raw in [("FP16",base)] + [("fake/"+a,f) for a,f in summary["fake_references"].items()]:
        if raw.get("status") != "completed":
            continue
        peak = max(t["memory"]["peak_allocated_bytes"] for t in raw["prefill"])
        lines.append(f"| {label} | {raw['memory']['resident_weight_bytes']/2**30:.3f} | {raw['resident_after_probe']['allocated_bytes']/2**30:.3f} | — | "
            f"{peak/2**30:.3f} | {raw['decode']['memory']['peak_allocated_bytes']/2**30:.3f} | {raw['decode']['kv_cache_final']['storage_bytes']/2**20:.3f} |")
    for r in good:
        peak = max(t["memory"]["peak_allocated_bytes"] for t in r["prefill"])
        lines.append(f"| {r['algorithm']}/{r['backend']} | {r['resident_weight_bytes']/2**30:.3f} | {r['resident_allocated_bytes']/2**30:.3f} | "
            f"{100*r['resident_saving_vs_fake']:.2f}% | {peak/2**30:.3f} | {r['decode']['memory']['peak_allocated_bytes']/2**30:.3f} | {r['decode']['kv_cache_final']['storage_bytes']/2**20:.3f} |")
    lines += ["", "## 解释与限制", "",
        "- numerical_warning 表示数值校验异常；延迟和比值仅为该异常路径的观测值，不能当作正确实现的有效性能收益。",
        "- PPL 是固定小样本的配对结果，不代表全量 WikiText-2 质量。各片段独立评分，累计 NLL 后计算总体 PPL。",
        "- 单层检查覆盖首层七类投影的 M=1、32、128、512；模型 probe 检查全部 252 个量化层执行。",
        "- 第三方算子保持原样。Marlin 的历史分块实验不复用；Nunchaku 使用项目原有每次至多 8 行的适配路径。",
        "- HF eager、主机调度及普通 KV cache 均在测量路径中；数据不等同于推理引擎的端到端生成吞吐。",
        "- 少量重复的 P90 只作粗略尾延迟观察。模型加载、准备和首次调用另行记录。",
        "- source.zip、source.diff、identity.json、samples.json、jobs/、logs/ 和逐配置 JSON 保存复现证据。"]
    failed = [r for r in summary["results"] if "error" in r]
    if failed:
        lines += ["", "## 执行失败", ""] + [f"- {r['algorithm']}/{r['backend']}: {r['error']}" for r in failed]
    (Path(output) / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    args = parser().parse_args()
    if args.sample_count < 1 or args.sample_length < 2 or args.warmup < 0 or args.repetitions < 1:
        raise ValueError("invalid sample or timing arguments")
    if min(args.lengths) < 1 or max(args.lengths) > args.sample_length:
        raise ValueError("prefill lengths must fit inside one saved sample")
    if args.decode_context < 1 or args.decode_steps < 1 or args.decode_context+args.decode_steps > args.sample_length:
        raise ValueError("decode context and steps must fit inside one saved sample")
    if not 1 <= args.probe_tokens <= args.sample_length:
        raise ValueError("invalid probe length")
    devices = [int(value.removeprefix("cuda:")) for value in args.devices]
    if len(set(devices)) != len(devices) or not set(devices) <= {g["index"] for g in query_gpus()}:
        raise ValueError("duplicate or unavailable GPU indices")
    kernel_diff = subprocess.check_output(["git", "diff", "HEAD", "--", "qslab/kernels/marlin", "qslab/kernels/w4a16"], cwd=ROOT, text=True)
    if kernel_diff:
        raise ValueError("operator source differs from HEAD; this benchmark requires original operators")
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H%M%S")
    args.output = (args.output or ROOT / "results/real_quant" / stamp).resolve()
    args.artifact_root = args.artifact_root.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "status.json"
    signal.signal(signal.SIGTERM, stop_children)
    signal.signal(signal.SIGINT, stop_children)
    write_json(state_path, {"status": "running", "stage": "identity", "pid": os.getpid(), "started_utc": utc_now()})
    print(f"OUTPUT {args.output}", flush=True)
    try:
        import torch
        token_ids = torch.load(args.data / "wikitext_test.pt", map_location="cpu", weights_only=False)["token_ids"]
        manifest = select_samples(token_ids, args.sample_count, args.sample_length, args.sample_seed)
        write_json(args.output / "samples.json", manifest)
        identity = build_identity(args.model, args.data)
        identity.pop("max_length", None)
        identity.pop("stride", None)
        identity.pop("use_cache", None)
        identity["protocol"] = {"name": "small_paired_v2", "sample_count": args.sample_count,
            "sample_length": args.sample_length, "sample_seed": args.sample_seed,
            "sample_token_sha256": manifest["token_sha256"], "scored_tokens": manifest["scored_tokens"],
            "prefill_lengths": list(args.lengths), "decode_context": args.decode_context,
            "decode_steps": args.decode_steps, "warmup": args.warmup,
            "repetitions": args.repetitions, "probe_tokens": args.probe_tokens}
        identity["identity"] = digest_json({k:v for k,v in identity.items() if k != "identity"})
        write_json(args.output / "identity.json", identity)
        write_json(args.output / "invocation.json", {"argv": sys.argv, "python": sys.executable,
            "conda_prefix": os.environ.get("CONDA_PREFIX"), "devices": devices,
            "measurement_gpu": devices[0], "algorithms": list(args.algorithms),
            "backends": list(args.backends), "protocol": identity["protocol"],
            "max_ppl_drift": args.max_ppl_drift, "artifact_root": str(args.artifact_root),
            "operators_match_HEAD": True})
        diff = subprocess.run(["git", "diff"], cwd=ROOT, capture_output=True, text=True)
        (args.output / "source.diff").write_text(diff.stdout)
        with zipfile.ZipFile(args.output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for path in source_files():
                archive.write(path, path.relative_to(ROOT))
            archive.write(ROOT / "benchmarks/real_quant/README.md", "benchmarks/real_quant/README.md")
        pairs = matrix(args.algorithms, args.backends)
        if not pairs:
            raise ValueError("no supported algorithm/backend pairs selected")
        algorithms = list(dict.fromkeys(a for a,_ in pairs))
        stages = [("prepare", [("prepare",a,None) for a in algorithms],devices),
                  ("reference", [("reference",a,None) for a in ("baseline",*algorithms)],devices[:1]),
                  ("real", [("real",a,b) for a,b in sorted(pairs,key=lambda p:p[1]=="qslab_w4a16")],devices[:1])]
        for phase,jobs,available in stages:
            write_json(state_path,{"status":"running","stage":phase,"pid":os.getpid(),"updated_utc":utc_now()})
            rows = run_phase(args,identity,jobs,available)
            if phase in ("prepare", "reference") and any(r["status"] != "completed" for r in rows):
                raise RuntimeError(f"{phase} failed; see jobs and logs")
        summary = aggregate(args,identity,pairs)
        write_json(state_path,{"status":summary["status"],"pid":os.getpid(),"finished_utc":utc_now()})
        print(f"COMPLETE {args.output / 'report.md'} status={summary['status']}",flush=True)
    except KeyboardInterrupt:
        write_json(state_path,{"status":"stopped","pid":os.getpid(),"finished_utc":utc_now()})
        raise
    except Exception as error:
        write_json(state_path,{"status":"failed","pid":os.getpid(),"finished_utc":utc_now(),"error":repr(error)})
        raise


if __name__ == "__main__":
    main()
