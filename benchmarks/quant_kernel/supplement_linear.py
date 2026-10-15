"""Only measure missing dense FP16/fake baselines; reuse historical kernel results."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmarks.quant_kernel.cases import ALGORITHMS, BACKENDS, CASES, GROUP_SIZES
from benchmarks.quant_kernel.metrics import measure
from benchmarks.quant_kernel.reference import compare, dequantize, reference_linear
from benchmarks.real_quant.common import (contended, file_sha256, query_gpus,
    selected_gpu, utc_now, write_json)
from qslab.compression.algorithms import QUANTIZERS


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--historical-summary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    original_bytes = args.historical_summary.read_bytes()
    old = json.loads(original_bytes)
    if old.get("performance_may_be_contended") or any(r["status"] != "pass" for r in old["results"]):
        raise ValueError("historical results must be the clean, passing kernel run")
    expected_cases = [(b,c["name"]) for b in BACKENDS for c in CASES]
    if [(r["backend"],r["case"]) for r in old["results"]] != expected_cases:
        raise ValueError("historical RNG sequence differs from the supported full matrix")
    if old["torch"] != torch.__version__ or old["cuda"] != torch.version.cuda:
        raise ValueError("PyTorch/CUDA differs from historical environment")
    before = query_gpus()
    gpu = selected_gpu(before)
    if contended(gpu):
        raise ValueError(f"selected GPU is occupied: {gpu}")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    if gpu["name"] != old["gpu"]:
        raise ValueError("GPU model differs from historical environment")
    torch.manual_seed(old["seed"])
    torch.cuda.manual_seed_all(old["seed"])
    write_json(args.output / "status.json", {"status":"running", "pid":os.getpid(), "started_utc":utc_now()})
    rows = []
    # Preserve every historical CPU RNG draw, including draws for other backends.
    # No quantized adapter is imported or invoked by this supplement.
    for backend in BACKENDS:
        for case in CASES:
            m,n,k = case["m"],case["n"],case["k"]
            weight = torch.randn(n,k,dtype=torch.float16)*0.02
            x_cpu = torch.randn(m,k,dtype=torch.float16)*0.1
            act_stat = torch.rand(k,dtype=torch.float16)+0.1
            algorithm,group = ALGORITHMS[backend],GROUP_SIZES[backend]
            logical,act_scale = QUANTIZERS[algorithm](weight,group_size=group,
                                                     activation_absmean=act_stat)
            x,w_fp16 = x_cpu.to(device),weight.to(device)
            restored = dequantize(logical,device)
            if act_scale is not None:
                restored = restored / act_scale.to(device=device,dtype=torch.float32).reshape(1,-1)
            w_fake = restored.to(torch.float16).contiguous()
            del restored

            def fp16_run(inputs):
                return F.linear(inputs,w_fp16)

            def fake_run(inputs):
                return F.linear(inputs,w_fake)

            # References and weight reconstruction are outside the timer.
            fp16_errors = compare(fp16_run(x),F.linear(x.float(),w_fp16.float()))
            fake_errors = compare(fake_run(x),reference_linear(x,logical,act_scale))
            if any(e["relative_l2"] >= 5e-3 or e["cosine_similarity"] <= 0.9999
                   for e in (fp16_errors,fake_errors)):
                raise ValueError("dense baseline reconstruction failed accuracy check")
            historical = next(r for r in old["results"] if r["backend"]==backend and r["case"]==case["name"])
            baseline = measure(fp16_run,x,old["warmup"],old["repetitions"])
            fake = measure(fake_run,x,old["warmup"],old["repetitions"])
            row = {"backend":backend,"algorithm":algorithm,"case":case["name"],"m":m,"n":n,"k":k,
                "group_size":group,"status":"pass","fp16":baseline,"fake":fake,
                "fp16_accuracy":fp16_errors,"fake_accuracy":fake_errors,
                "historical_kernel_performance":historical["performance"],
                "kernel_speedup_vs_fp16":baseline["latency_median_ms"]/historical["performance"]["latency_median_ms"],
                "kernel_speedup_vs_fake":fake["latency_median_ms"]/historical["performance"]["latency_median_ms"],
                "reconstructed_tensor_sha256":{"input":tensor_sha256(x_cpu),"original_weight":tensor_sha256(weight),
                    "logical_codes":tensor_sha256(logical.values),"fake_weight":tensor_sha256(w_fake)}}
            rows.append(row)
            write_json(args.output / "partial.json",rows)
            print(json.dumps(row),flush=True)
            del x,w_fp16,w_fake,x_cpu,weight,act_stat,logical,act_scale
            torch.cuda.empty_cache()
    if args.historical_summary.read_bytes() != original_bytes:
        raise AssertionError("historical kernel result was modified")
    summary = {"created_utc":utc_now(),"status":"completed","kind":"dense_baseline_supplement",
        "historical_summary":str(args.historical_summary),"historical_summary_sha256":hashlib.sha256(original_bytes).hexdigest(),
        "historical_measured_utc":old["created_utc"],"historical_kernel_reexecuted":False,
        "gpu":gpu,"gpu_state_before":before,"gpu_state_after":query_gpus(),
        "performance_may_be_contended":False,"torch":torch.__version__,"cuda":torch.version.cuda,
        "python":platform.python_version(),"seed":old["seed"],"warmup":old["warmup"],"repetitions":old["repetitions"],
        "scope":"synchronized host wall time; historical packed kernels vs newly measured dense F.linear",
        "fake_definition":"logical dequantization with AWQ input scale folded into FP16 weights before timing",
        "rng_replay":"CPU weight/input/activation draws in the original full backend/case order; historical input tensors were not archived",
        "script_sha256":file_sha256(Path(__file__)),"results":rows}
    write_json(args.output / "summary.json",summary)
    write_report(args.output,summary)
    write_json(args.output / "status.json",{"status":"completed","pid":os.getpid(),"finished_utc":utc_now()})
    print(f"COMPLETE {args.output / 'report.md'}",flush=True)


def write_report(output,summary):
    lines=["# 量化算子：补测同形状 FP16 / Fake Linear", "",
        "仅新增 dense Linear 对照。历史三个量化后端的延迟直接读取归档，未重新执行。",
        f"历史量化测量时间：`{summary['historical_measured_utc']}`；补测时间：`{summary['created_utc']}`（UTC）。",
        f"硬件：{summary['gpu']['name']}，物理 GPU {summary['gpu']['index']}；预热 {summary['warmup']} 次，正式 {summary['repetitions']} 次。",
        "计时复用原 measure()：包含 Python 调用、CUDA 执行及同步的主机墙钟；表中延迟为中位数，单位 µs。",
        "Fake 为同份逻辑量化权重重建的 FP16 F.linear，AWQ 缩放提前折入权重，计时不含量化或反量化。",
        "", "| 后端 | (M,N,K) | FP16 µs | Fake µs | 历史量化 µs | 量化相对 FP16 | 量化相对 Fake |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for r in summary["results"]:
        lines.append(f"| {r['backend']} | ({r['m']},{r['n']},{r['k']}) | {1000*r['fp16']['latency_median_ms']:.3f} | "
            f"{1000*r['fake']['latency_median_ms']:.3f} | {1000*r['historical_kernel_performance']['latency_median_ms']:.3f} | "
            f"{r['kernel_speedup_vs_fp16']:.3f}× | {r['kernel_speedup_vs_fake']:.3f}× |")
    lines += ["", "## 解释", "",
        "- 加速比=浮点延迟/量化延迟，大于 1 表示量化更快，小于 1 表示量化更慢。",
        "- 每个后端历史上使用独立合成输入；补测重放原 seed、随机数生成顺序、算法和分组，逐后端配对。",
        "- 历史输入张量未归档，无法用历史张量哈希做直接核对；本轮保存重建张量哈希。",
        "- 权重、输入形状和 dtype 相同；FP16 使用原始权重，Fake 使用该后端的量化重建权重。",
        "- 历史和补测来自不同时段，因此加速比是跨运行对照，不能排除时钟、温度、主机调度的差异。",
        "- 仅适用于原先四种小矩阵，不能推导真实模型长输入性能或纯 CUDA kernel 时间。",
        "- 两种新增 dense 路径都与 FP32 reference 对照；原量化算子正确性沿用历史记录。"]
    (Path(output)/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
