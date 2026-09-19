# 07 — 前缀缓存 TTFT（M9，物化路线修复后）

## 证据链

底稿 results/m9_prefix_cache.txt（3800-token 前缀 + 11 后缀，max_tokens=1 取首 token 墙钟）：

| 模型 | 冷（turn-1） | 命中（turn-2） | 加速 |
|---|---|---|---|
| 1.7B（M9） | 136.2ms | 49.9ms | **2.73×** |
| 8B（M9，当时只在 notes 里口录） | 558.3ms | 63.0ms | 8.86× |
| 8B（M10 重跑落盘） | 555.9ms | 62.3ms | **8.92×** |

M10 重跑与 M9 记录差 0.4% / 1.1%，两段原始输出（含命令与 `[kvalloc]` 行）都在同一
底稿里。重跑顺带确认两件此前没写清的事：

* 本 bench **没有 W4 开关**，走的是 fp16 权重 + int4 KV，测的是 KV 物化路线而不是
  权重量化路线；
* 8B 因此需要 `UTIL=0.9` 才放得下（`peak=16.91GiB`，KV 池只剩 `budget=3.69GiB`）。
  默认 `UTIL=0.6` 会在分配阶段直接 OOM——想省显存请先看 `02-w4-quant` 的常驻底稿。

背景：M8 发现静默错答 bug（命中前缀只在 int4 池、prefill 不读池），曾禁用；
M9 物化路线修复后默认开（ENABLE_PREFIX_CACHE=True）。正确性由 tests/test_prefix_cache.py
锁定（hit==hit 精确确定、命中确实发生、共享前缀+新后缀、chunked prefill 同路径）。

## 复现

```bash
python benchmarks/07-prefix-cache/bench_prefix_cache.py                       # 1.7B
MODEL=models/Qwen3-8B UTIL=0.9 python benchmarks/07-prefix-cache/bench_prefix_cache.py  # 8B
# CTX= 换前缀长度；需要 results/smooth_kv4_*.pt
```
