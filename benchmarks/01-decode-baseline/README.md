# 01 — decode 吞吐基线 + 长上下文检索（M0/M2/M4）

## 证据链

| 数字 | 含义 | 底稿 |
|---|---|---|
| 42.08 tok/s（median, std 3.07） | 1.7B FP16 引擎基线，512in/128dec | results/bench_throughput_Qwen3-1.7B_20260914_211828.json |
| 42.61 tok/s | 次日复测一致 | results/bench_throughput_Qwen3-1.7B_20260915_110626.json |
| 148.7 tok/s（graph）/ 22.8（eager） | 1.7B 新 runtime（M8，CUDA Graph 6.5×） | notes/M8-整合.md §五 |
| NIAH @1K fp16/kv4 双 100%；@2K+ 双 0% | 1.7B 检索上限 1K-2K，KV4 与 FP16 同步衰减 | results/m2_niah.json |
| NIAH 32K fp16/kv4 双 100% | 8B 上 KV4 无损 | notes/M4-组合.md |

结论：42 tok/s 是全部加速工作的原点；"KV4 与 FP16 同步衰减"本身是量化无损性的强证据。

## 复现

两个脚本已于 2026-09-20 随旧引擎从 main 摘除，完整保留在存档分支
`archive/legacy-engine`（= 标签 `legacy-engine-final` = `beaa88a`）：

```bash
git checkout archive/legacy-engine
python benchmarks/01-decode-baseline/bench_throughput.py --model models/Qwen3-1.7B
python benchmarks/01-decode-baseline/bench_niah.py
```

新 runtime 的吞吐口径走 `benchmarks/09-batch-throughput/bench_batch.py`。
