# 03 — PPL 主线 + KV4 量化代价（M0-M4 / M8）

## 证据链

PPL 主线（引擎实测，WikiText-2 raw）：

| 配置 | 1.7B | 8B | 底稿 |
|---|---|---|---|
| FP16 | 26.48 | 16.07 | results/ppl_Qwen3-1.7B_*.json / m4_matrix_fp16.json |
| W4 (AWQ) | 27.72 (+1.24) | 17.31 (+1.24) | notes/M1a / m4_matrix_w4.json |
| + KV4 | 27.80 | 17.70 | m4_matrix_w4kv4.json |

KV4 代价归因（M8，`bench_ppl_kv.py` 在 HF 上模拟方案、绕开 runtime）：

| 配置（8B，10 docs） | PPL | 底稿 |
|---|---|---|
| fp16 | 16.059 | results/m8_kv4_ppl.json |
| 仅 K int4（SmoothAttention） | 16.202 (+0.143) | 同上 |
| 仅 V int4 | 16.146 (+0.086) | 同上 |
| KV4 | 16.428 (**+0.369, +2.3%**) | 同上 |

方法要点（复现时不可改，否则测的是别的东西）：
- **拦截点必须 post-RoPE**（挂 k_proj 会量化到 norm/rope 之前——第一次就这么错，PPL 2077）
- K=静态 per-channel（SmoothAttention λ）、V=动态 per-token g64
- 1.7B KV4 代价 +1.098（+4.1%），更小模型校准统计天然吃亏
- 自校验：8B fp16 基线 16.059 与 M4 矩阵的 16.07 吻合

## 复现

```bash
python benchmarks/03-kv4-quality/bench_ppl.py --model models/Qwen3-8B --max-docs 16
python benchmarks/03-kv4-quality/bench_ppl_kv.py    # 纯 HF，不依赖 runtime
```
