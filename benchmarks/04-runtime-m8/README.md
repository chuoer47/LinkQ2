# 04 — 8B 四方矩阵 + paged 验收 + runtime PPL（M4/M7/M8）

## 证据链

**8B 四方矩阵**（M4，旧引擎，底稿 results/m4_matrix_*.json）：

| 模式 | tok/s | PPL | KV mem@512 | 权重 |
|---|---|---|---|---|
| FP16 | 42.4 | 16.07 | 94.2 MB | 16.4 GB |
| W4A16 | 14.2 | 17.31 | 94.2 MB | 2.49 GB (6.6×) |
| W4+KV4 | 9.0 | 17.70 | 27.0 MB (3.5×) | 2.49 GB |
| W4+KV4+Spec | 10.8 | — | 27.0 MB | 2.49+1.2 GB |

**M7 paged 验收**（底稿 results/m7_acceptance.json）：
paged vs dense KV4 PPL 差 **-0.20**（噪声边缘）；@8K 峰值显存 **24.4→18.4 GB**
（O(context) fp16 副本消失实证）。

**M8 runtime PPL**（底稿 results/m8_kv4_ppl.json + bench_ppl_runtime.py）：
新 runtime 真实 decode 路径的 teacher-forcing PPL；KV4 池显存 6.92 GiB vs fp16 13.84（2×）。

⚠ W4=14.2 的短板已被 M5（Marlin）与 M9（CROSSOVER_M=0）两轮修正，
现行主线 8B decode = 97.2 tok/s（见 05-spec-ngram 底稿的 off 行）。

## 复现

```bash
# 四方矩阵，每次一个模式（显存原因）
python benchmarks/04-runtime-m8/bench_engine_matrix.py fp16
python benchmarks/04-runtime-m8/bench_engine_matrix.py w4kv4 --ppl

# M7 验收（8B paged vs dense）
python benchmarks/04-runtime-m8/m7_acceptance.py

# 新 runtime decode 路径 PPL（teacher forcing）
python benchmarks/04-runtime-m8/bench_ppl_runtime.py
```
