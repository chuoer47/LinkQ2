"""Sequential, isolated workers; SIGINT/SIGTERM stop the owned child process."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

from benchmarks.real_quant.common import write_json, file_sha256, utc_now
from benchmarks.real_quant.config import ROOT, ARTIFACTS


def summarize(output):
    cases = [json.loads(p.read_text()) for p in sorted((output / "cases").glob("*.json"))]
    lookup = {(r["dtype"], r["algorithm"], r["mode"], r["accumulation"]): r for r in cases}
    rows = []
    for r in cases:
        if r["status"] != "completed":
            continue
        dense = lookup.get((r["dtype"], "baseline", "dense", "fp32"))
        fake = lookup.get((r["dtype"], r["algorithm"], "fake", "fp32"))
        row = {"key": r["key"], "dtype": r["dtype"], "algorithm": r["algorithm"],
            "dispatch": r.get("dispatch", "module"), "accumulation": r["accumulation"],
            "numerical_valid": all(c["passed"] for c in r.get("linear_checks", [])),
            "eligible_for_speed_ranking": r.get("eligible_for_speed_ranking", True),
            "mode": r["mode"], "ppl": r["quality"]["ppl"],
            "resident_gib": r["resident_after_probe"]["allocated_bytes"] / 2**30,
            "decode_ms": r["decode"]["latency_median_ms"], "decode_tokens_s": r["decode"]["tokens_per_second"],
            "decode_peak_gib": r["decode"]["memory"]["peak_allocated_bytes"] / 2**30,
            "quality_peak_gib": r["quality_memory"]["peak_allocated_bytes"] / 2**30,
            "linear_checks_passed": sum(c["passed"] for c in r.get("linear_checks", [])),
            "linear_checks_total": len(r.get("linear_checks", []))}
        for p in r["prefill"]:
            row[f"prefill_{p['sequence_length']}_ms"] = p["latency_median_ms"]
            row[f"prefill_{p['sequence_length']}_peak_gib"] = p["memory"]["peak_allocated_bytes"] / 2**30
        if dense and dense["status"] == "completed":
            row.update(decode_speedup_vs_dense=dense["decode"]["latency_median_ms"]/row["decode_ms"],
                resident_saving_percent=100*(1-r["resident_after_probe"]["allocated_bytes"]/dense["resident_after_probe"]["allocated_bytes"]),
                ppl_delta_vs_dense_percent=100*(row["ppl"]/dense["quality"]["ppl"]-1))
            for p in dense["prefill"]:
                s = p["sequence_length"]
                row[f"prefill_{s}_speedup_vs_dense"] = p["latency_median_ms"] / row[f"prefill_{s}_ms"]
        if fake and fake["status"] == "completed":
            row["ppl_delta_vs_fake_percent"] = 100*(row["ppl"]/fake["quality"]["ppl"]-1)
        rows.append(row)
    write_json(output / "summary.json", {"cases": rows, "failed": [r for r in cases if r["status"] != "completed"]})
    if rows:
        import csv
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# GemLite 融合 W4A16 实测", "",
        "Qwen3-8B；单张 RTX 4090 D；HF eager，不接推理引擎。重打包已有量化整数、scale、zero，未重新校准或量化。", "",
        "固定 8×512 token，4088 个 PPL 计分目标。prefill 为完整 forward，包含 lm_head；decode 为 128 token 上下文后 32 步固定输入，普通 HF KV cache。模型计时使用同步主机墙钟，预热 2 次、重复 5 次；编译和自动调优在计时前完成。", "",
        "FP16/BF16 表示激活、输出和其余模型权重类型。配置名中的 gemlite_fp32/fp16 是传入 GemLite 的 acc_dtype 请求，GEMM/SplitK 按此设置；M=1 的 GEMV_REVSPLITK 上游实现会覆盖为输入 dtype，并使用输出 dtype 的原子归约，因此不能称作全路径 FP32 累加。量化 scale 维持原 artifact 的 FP16 精度。配置名以 direct 结尾时，适配器直接调用同一上游 kernel.forward；其余使用上游模块及 torch.library 调度。完整模型均未使用 CUDA graph 或 torch.compile。", "",
        "| 配置 | PPL | 相对 fake PPL | 常驻 GiB | prefill128 ms | prefill512 ms | decode ms/token | decode 加速比 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        delta = f"{r['ppl_delta_vs_fake_percent']:+.4f}%" if "ppl_delta_vs_fake_percent" in r else "—"
        speed = f"{r['decode_speedup_vs_dense']:.3f}×" if "decode_speedup_vs_dense" in r else "—"
        lines.append(f"| {r['key']} | {r['ppl']:.6f} | {delta} | {r['resident_gib']:.3f} | {r['prefill_128_ms']:.3f} | {r['prefill_512_ms']:.3f} | {r['decode_ms']:.3f} | {speed} |")
    lines.extend(["", "加速比 = 同精度、同次实验 dense 延迟 / real 延迟；大于 1 才表示加速。常驻显存为模型加载且执行过 probe 后，清除临时张量的 PyTorch allocated；各阶段峰值另见 cases/*.json，reserved 不作为权重显存收益。", "",
        "单算子以 Qwen 第一层四类真实矩阵形状及 M=1/32/128/512 测量，同输入的原始浮点权重、fake 权重和 GemLite 同场比较。host 为同步调用延迟；CUDA graph + event 去掉 CPU 发射空隙，代表热缓存、图回放条件，不能替代完整模型 decode。完整模型没有使用 CUDA graph。", "",
        "fake 将 AWQ 输入缩放提前折入浮点权重，real 在 forward 中除输入 scale，两者存在浮点舍入顺序差异。BF16 fake 从同一逻辑权重直接反量化并转换 BF16。PPL 仅用于本次配对比较，不能等同完整测试集指标。", "",
        "单算子门槛：FP16 relative L2<0.005；BF16<0.01；cosine>0.9999；不通过则停止该配置，不修改上游算子。每个 real 配置验证 28 个投影/形状，probe 验证全部 252 个替换层确实执行。", "",
        "运行来源、完整命令、环境、原始计时、每段 PPL、配对 logits 差异、KV cache 精度、GPU 状态、峰值显存均保存在 manifest.json、jobs.json、cases/ 和 logs/；源码在 source.zip。"])
    lines += ["", "## 融合路径相对同精度原始模型", "",
        "| 配置 | prefill128 加速比 | prefill512 加速比 | decode 加速比 | 常驻节省 | prefill512 峰值 GiB | decode 峰值 GiB | 数值检查 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["mode"] == "gemlite" and "decode_speedup_vs_dense" in r:
            lines.append(f"| {r['key']} | {r['prefill_128_speedup_vs_dense']:.3f}× | {r['prefill_512_speedup_vs_dense']:.3f}× | {r['decode_speedup_vs_dense']:.3f}× | {r['resident_saving_percent']:.2f}% | {r['prefill_512_peak_gib']:.3f} | {r['decode_peak_gib']:.3f} | {r['linear_checks_passed']}/{r['linear_checks_total']} |")
    for case in cases:
        if not case.get("microbenchmark"):
            continue
        lines += ["", f"## 单算子 {case['key']}", "",
            "| M,N,K | 浮点 host μs | Fake host μs | 融合 host μs | host 加速比 | 浮点 GPU μs | Fake GPU μs | 融合 GPU μs | GPU 加速比 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for m in case["microbenchmark"]:
            t = m["measurements"]
            lines.append(f"| {m['m']},{m['n']},{m['k']} | {t['dense']['host_median_ms']*1000:.2f} | {t['fake']['host_median_ms']*1000:.2f} | {t['gemlite']['host_median_ms']*1000:.2f} | {m['speedup_vs_dense_host']:.3f}× | {t['dense']['cuda_graph_median_ms']*1000:.2f} | {t['fake']['cuda_graph_median_ms']*1000:.2f} | {t['gemlite']['cuda_graph_median_ms']*1000:.2f} | {m['speedup_vs_dense_gpu']:.3f}× |")
    failed = [r for r in cases if r["status"] != "completed"]
    if failed:
        lines += ["", "## 未完成配置", ""] + [f"- {r['key']}：{r.get('error',r['status'])}" for r in failed]
    warning_rows = [r for r in rows if not r["numerical_valid"]]
    if warning_rows:
        lines += ["", "## 数值告警的诊断测量", "",
            "diagnostic 配置保留原数值门槛及失败记录，仅补测 PPL 与延迟以观察误差影响，不参与通过数值检查的性能结论。"]
        for r in warning_rows:
            lines.append(f"- {r['key']}：检查 {r['linear_checks_passed']}/{r['linear_checks_total']} 通过；PPL={r['ppl']:.6f}，相对 fake {r.get('ppl_delta_vs_fake_percent',0):+.4f}%。")
    (output / "README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", default="1")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--previous", type=Path, default=ROOT / "results/real_quant/2026-10-13_221713-small-v2")
    p.add_argument("--summarize-only", action="store_true")
    args = p.parse_args()
    output = args.output.resolve()
    if args.summarize_only:
        summarize(output)
        return
    output.mkdir(parents=True, exist_ok=True)
    for name in ("cases", "logs", "paired"):
        (output / name).mkdir(exist_ok=True)
    if (output / "manifest.json").exists():
        raise ValueError("use a new output directory; existing runs are immutable")
    old = json.loads((args.previous / "identity.json").read_text())
    shutil.copyfile(args.previous / "samples.json", output / "samples.json")
    samples = json.loads((output / "samples.json").read_text())
    artifact_root = ARTIFACTS / old["artifact_identity"]
    original_diff = subprocess.check_output(["git", "diff", "HEAD", "--", "qslab/kernels/marlin", "qslab/kernels/w4a16"], cwd=ROOT, text=True)
    if original_diff.strip():
        raise RuntimeError("original third-party operator tree has changes")
    algorithms = ("rtn", "rtn_clip", "awq", "nunchaku_awq")
    commands = [("fp16", "baseline", "dense"), ("bf16", "baseline", "dense")]
    commands += [("fp16", a, mode) for a in algorithms for mode in ("fake", "gemlite")]
    commands += [("bf16", "rtn", mode) for mode in ("fake", "gemlite")]
    manifest = {"started_utc": utc_now(), "python": sys.executable,
        "model": old["model"], "artifact_root": str(artifact_root),
        "artifact_identity": old["artifact_identity"], "previous": str(args.previous),
        "sample_token_sha256": samples["token_sha256"], "gpu": args.gpu,
        "gemlite_version": importlib.metadata.version("gemlite"),
        "installed_package_sources": json.loads((ROOT / ".cache/gemlite/packages048.json").read_text()),
        "environment_isolation": "project venv --system-site-packages; GemLite and Triton installed only in venv with --no-deps --no-build-isolation; torch inherited, torch.compile/inductor unused",
        "matrix": commands, "status": "verifying", "original_operator_diff": original_diff}
    write_json(output / "manifest.json", manifest)
    sources = list((ROOT / "benchmarks/fused_quant").glob("*.py")) + [ROOT / "benchmarks/fused_quant/README.md"]
    sources += list((ROOT / "benchmarks/real_quant").glob("*.py"))
    sources += [ROOT / "benchmarks/quant_kernel/reference.py"]
    manifest["source_hashes"] = {str(f.relative_to(ROOT)): file_sha256(f) for f in sources}
    packages = ("torch", "triton", "transformers", "safetensors", "gemlite")
    manifest["environment"] = {n: importlib.metadata.version(n) for n in packages}
    manifest["pip_freeze"] = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines()
    base_python = "/home/amax01/miniconda3/envs/qslab/bin/python"
    base_check = "import importlib.metadata,json; print(json.dumps({n:importlib.metadata.version(n) for n in ('torch','triton','transformers','safetensors')}))"
    manifest["original_conda_environment"] = json.loads(subprocess.check_output([base_python, "-c", base_check], text=True))
    with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for f in sources:
            archive.write(f, str(f.relative_to(ROOT)))
    verified = {}
    for algorithm in algorithms:
        directory = artifact_root / algorithm
        complete = json.loads((directory / "complete.json").read_text())
        assert complete["artifact_identity"] == old["artifact_identity"]
        for name in ("logical/manifest.json", "logical/tensors.safetensors"):
            sha = file_sha256(directory / name)
            assert sha == complete["files"][name]["sha256"], (algorithm, name)
            verified[f"{algorithm}/{name}"] = sha
        print(f"VERIFIED {algorithm}", flush=True)
    manifest.update(status="running", verified_artifacts=verified)
    write_json(output / "manifest.json", manifest)
    child = None
    def stop(signum, frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
        manifest.update(status="stopped", finished_utc=utc_now())
        write_json(output / "manifest.json", manifest)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    jobs = []
    for dtype, algorithm, mode in commands:
        key = f"{dtype}_{algorithm}_{mode}_fp32"
        command = [sys.executable, "-u", "-m", "benchmarks.fused_quant.worker", "--output", str(output),
            "--dtype", dtype, "--algorithm", algorithm, "--mode", mode]
        if mode == "gemlite" and algorithm == "rtn":
            command.append("--micro")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, TOKENIZERS_PARALLELISM="false")
        with (output / "logs" / (key + ".log")).open("w") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            job = {"key": key, "pid": child.pid, "command": command,
                   "CUDA_VISIBLE_DEVICES": args.gpu, "started_utc": utc_now()}
            jobs.append(job)
            write_json(output / "jobs.json", jobs)
            print(f"START {key} pid={child.pid}", flush=True)
            code = child.wait()
            job.update(returncode=code, finished_utc=utc_now())
            write_json(output / "jobs.json", jobs)
        summarize(output)
        print(f"END {key} returncode={code}", flush=True)
        if code:
            # Do not proceed with a real path if its reference failed.
            manifest.update(status="failed", failed_job=key, finished_utc=utc_now())
            write_json(output / "manifest.json", manifest)
            raise SystemExit(code)
    manifest.update(status="completed", finished_utc=utc_now())
    write_json(output / "manifest.json", manifest)
    print(f"ALL COMPLETE {output}", flush=True)


if __name__ == "__main__":
    main()
