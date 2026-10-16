# 混合 Chunked Prefill 实测

2026-10-16，RTX 4090 D GPU 1，Qwen3-8B，代码 `a00597e`。来源 nano-vLLM PR #276 的 `dbf934f`，尚未合并 main；适配实现为 `696c14a`。本轮不修改 GemLite/KV4 kernel 计算。

小规模三请求：一条短请求先执行两步，再加入一条 512-token 请求与另一条短请求；合计输出 32 token，预算 64 token/step，最多 4 条序列，KV 容量 2048 token。两种策略都启用纯 decode graph；含 prefill 的批次使用 eager varlen。各配置独立进程，完整预热一轮、测量三轮。

| 权重 / KV | 指标 | prefill-first | mixed |
|---|---|---:|---:|
| FP16 / KV16 | 原短请求 decode 推迟步数 | 9 | 0 |
| FP16 / KV16 | 原短请求最大 token 间隔 ms | 347.247 | 38.111 |
| FP16 / KV16 | 总耗时 ms | 659.499 | 537.115 |
| FP16 / KV16 | 输出 token/s | 48.52 | 59.58 |
| FP16 / KV16 | 长请求 TTFT ms | 289.553 | 338.834 |
| W4 / KV4 | 原短请求 decode 推迟步数 | 9 | 0 |
| W4 / KV4 | 原短请求最大 token 间隔 ms | 868.491 | 102.273 |
| W4 / KV4 | 总耗时 ms | 1116.264 | 1095.705 |
| W4 / KV4 | 输出 token/s | 28.67 | 29.20 |
| W4 / KV4 | 长请求 TTFT ms | 768.469 | 900.159 |

每轮 mixed 都实际执行 9 步混合批次，原短请求没有因 prefill 被推迟，始终遵守 token 预算；纯 decode graph replay、完成后的页释放、关闭后的显存基线检查通过。

结果体现取舍：现有 decode 的最大停顿显著缩短，总耗时下降；长请求 TTFT 增加，不能宣称所有延迟都改善。W4+KV4 在混合批仍需要展开缓存前缀，这一成本未做新算子优化。

49 项回归通过：覆盖 decode 优先及预算、共享前缀 watermark、中间分块跳过 lm_head/采样、FP16 eager/graph 下的分块/混合/HF 对照、KV4 混合 decoder 当前 token 的量化语义，以及生命周期、已有 KV4 算子与采样接受规则。测试日志在服务器仓库外 `../linkq2-chunked-20261016/tests.log`。

原始运行数据、每步请求/分块/采样标志、输出 token 与时间记录见 `results/chunked_prefill/2026-10-16/`。三请求小实验不构成高并发、长上下文或显存紧张时抢占策略的验收；投机推理不与本轮混合策略组合。
