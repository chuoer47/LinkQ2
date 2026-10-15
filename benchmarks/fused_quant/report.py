"""Compact Chinese findings, derived only from saved measurements."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    output = args.output
    summary = json.loads((output / "summary.json").read_text())
    rows = {r["key"]: r for r in summary["cases"]}
    valid = [r for r in rows.values() if r["mode"] == "gemlite" and
             r["numerical_valid"] and r["eligible_for_speed_ranking"]]
    assert len(valid) == 10
    assert all(r["linear_checks_passed"] == 28 for r in valid)
    assert all(r["decode_speedup_vs_dense"] < 1 for r in valid)
    lines = ["# 成熟融合 W4A16 实测结论", "",
        "**本次 GemLite 0.4.8 在普通 HF eager 路径上，显存收益明确，未获得端到端加速。** 单算子 GPU 测量有收益，但主机调用开销及部分 GEMM 性能限制了完整模型速度。此结论仅覆盖本次版本、接入方式和测试形状，不代表最新版或编译/图执行后的上限。", "",
        "环境：单张 RTX 4090 D，Qwen3-8B，PyTorch 2.5.1+cu124，独立项目 venv 中 GemLite 0.4.8 / Triton 3.3.0。不接推理引擎，完整模型不使用 torch.compile 或 CUDA graph；单算子另测 CUDA graph/event。固定同一份 WikiText-2 的 8×512 token，4088 个 PPL 计分目标。", "",
        "## RTN 与浮点基线", "",
        "| 配置 | 常驻 GiB | prefill128 ms | prefill512 ms | decode ms/token | 相对同精度浮点 decode 加速比 | PPL |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    keys = ["fp16_baseline_dense_fp32", "fp16_rtn_fake_fp32", "fp16_rtn_gemlite_fp32",
            "fp16_rtn_gemlite_fp32_direct", "bf16_baseline_dense_fp32", "bf16_rtn_fake_fp32",
            "bf16_rtn_gemlite_fp32", "bf16_rtn_gemlite_fp32_direct"]
    for key in keys:
        r = rows[key]
        lines.append(f"| {key} | {r['resident_gib']:.3f} | {r['prefill_128_ms']:.3f} | {r['prefill_512_ms']:.3f} | {r['decode_ms']:.3f} | {r['decode_speedup_vs_dense']:.3f}× | {r['ppl']:.6f} |")
    reference = rows["fp16_baseline_dense_fp32"]
    real = rows["fp16_rtn_gemlite_fp32_direct"]
    lines += ["", f"RTN 常驻显存节省 {real['resident_saving_percent']:.2f}%。prefill512 峰值 allocated 从 {reference['prefill_512_peak_gib']:.3f} GiB 降至 {real['prefill_512_peak_gib']:.3f} GiB；decode 峰值从 {reference['decode_peak_gib']:.3f} GiB 降至 {real['decode_peak_gib']:.3f} GiB。fake 的权重已离线展开为浮点，常驻显存与浮点模型相同。", "",
        "module 是原版 GemLite 模块入口；direct 是本项目适配器直接调用同一个原版 kernel.forward。direct 没有修改上游数学运算，并逐项与 module 校验。加速比 = 同精度浮点延迟 / 实测延迟，大于 1 才表示加速。", "",
        "## 四种算法的直接调用路径（FP16 输入/输出）", "",
        "| 算法 | fake PPL | real PPL | 相对 fake PPL | 常驻节省 | prefill128 加速比 | prefill512 加速比 | decode 加速比 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for algorithm in ("rtn", "rtn_clip", "awq", "nunchaku_awq"):
        r = rows[f"fp16_{algorithm}_gemlite_fp32_direct"]
        fake = rows[f"fp16_{algorithm}_fake_fp32"]
        lines.append(f"| {algorithm} | {fake['ppl']:.6f} | {r['ppl']:.6f} | {r['ppl_delta_vs_fake_percent']:+.5f}% | {r['resident_saving_percent']:.2f}% | {r['prefill_128_speedup_vs_dense']:.3f}× | {r['prefill_512_speedup_vs_dense']:.3f}× | {r['decode_speedup_vs_dense']:.3f}× |")
    max_drift = max(abs(r["ppl_delta_vs_fake_percent"]) for r in valid)
    lines += ["", f"10 个有效 real 配置均通过 28/28 个投影数值检查，全部 252 个替换层在 probe 中执行；相对配对 fake 的最大 PPL 差异为 {max_drift:.5f}%。这表示额外算子误差很小，并不消除量化算法相对原始模型的 PPL 损失。", "",
        "## 单算子 GPU 与主机延迟", "",
        "RTN、FP16 输入输出、M=1；四类 Qwen 实际矩阵。GPU 是预热后的 CUDA graph/event 单次均摊延迟，主机是普通调用加同步，条件不同，不能相互替代。", "",
        "| N,K | 浮点 GPU μs | 融合 GPU μs | GPU 加速比 | 浮点 host μs | 融合 host μs | host 加速比 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    case = json.loads((output / "cases/fp16_rtn_gemlite_fp32_direct.json").read_text())
    for m in case["microbenchmark"]:
        if m["m"] != 1:
            continue
        t = m["measurements"]
        lines.append(f"| {m['n']},{m['k']} | {t['dense']['cuda_graph_median_ms']*1000:.2f} | {t['gemlite']['cuda_graph_median_ms']*1000:.2f} | {m['speedup_vs_dense_gpu']:.3f}× | {t['dense']['host_median_ms']*1000:.2f} | {t['gemlite']['host_median_ms']*1000:.2f} | {m['speedup_vs_dense_host']:.3f}× |")
    lines += ["", "M=32/128/512 的完整同形状 FP16/BF16、fake、融合算子对照见 README.md 和各 case 的 microbenchmark；部分 prefill GEMM 的 GPU 延迟也慢于浮点。", "",
        "## FP16 累加诊断", ""]
    key = "fp16_rtn_gemlite_fp16_direct_diagnostic"
    if key in rows:
        r = rows[key]
        lines += [f"down_proj 在 M=128/512 时 relative L2 约 0.00576，超过预设 0.005，检查通过 {r['linear_checks_passed']}/{r['linear_checks_total']}。保持失败标记，单独补测得到 PPL {r['ppl']:.6f}（相对 fake {r['ppl_delta_vs_fake_percent']:+.5f}%）；prefill512 {r['prefill_512_ms']:.3f} ms，原始 FP16 为 {reference['prefill_512_ms']:.3f} ms；decode {r['decode_ms']:.3f} ms/token。此项不参与有效加速结论。"]
    lines += ["", "配置名 fp32/fp16 表示向 GemLite 请求的 acc_dtype。GEMM/SplitK 按请求使用，M=1 的上游 GEMV_REVSPLITK 会自行覆盖为输入 dtype，并采用输出 dtype 原子归约，不能称作全路径 FP32 累加。", "",
        "## 记录与复现", "",
        "中文运行说明：benchmarks/fused_quant/README.md；基础矩阵用 run.py，直接调用补充用 supplement.py。命令、PID、环境与原始数据见 manifest.json、jobs.json、supplement_manifest.json、diagnostic_manifest.json、cases/ 和 logs/。三个阶段源码分别保存在 source.zip、source_direct.zip、source_diagnostic.zip。失败适配器预检保留在 attempts/。", "",
        "新增依赖仅安装在项目 .venvs/gemlite-0.4.8，原 conda 包未更改；Torch 2.5.1 元数据固定 Triton 3.1，本 venv 为直接使用 GemLite 显式覆盖为 Triton 3.3，未使用 Inductor。上游包文件和原有 kernel 的哈希审计见 audit_before.json/audit_after.json。"]
    (output / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
