# KV4 修复与实测记录

日期：2026-10-16。RTX 4090D，项目 `.conda`：torch 2.7.1+cu126、Triton 3.3.1、FlashAttention 2.8.3。

## 修复和验证

1. Config 拒绝缺少或不存在的 smooth_kv 文件；加载校准时要求 lambda、kscale 的形状匹配模型，转为 FP16 后仍为有限正数，防止静态 K scale 为零时继续推理。
2. KV 池绑定 Attention 时无条件传入配置的 v_group，避免池布局与读写分组不一致；配置要求分组为至少 8 的 2 次幂且整除 head_dim。

15 项回归测试通过，包括非默认 v_group=32/128 的真实 GPU 写入与读取。
额外通过正式 LLM 入口加载 Qwen3-8B、v_group=128，核对全部 36 层的池布局并完成 8 token 生成。
该生成仅作主链路冒烟检查，不作为精度指标。

## 测量范围

固定原始 Qwen3-8B 的 FP16 权重，使用自研运行时模型；两种 KV 使用相同 SmoothAttention。
从独立 C4 训练片段 8×512 token 重新生成校准参数，alpha=0.5。
评测为 WikiText-2 固定 8×512 token，共 4088 个预测目标。

每段只有第一个输入 token 走 prefill，后续按 32 query 因果分块执行；KV4 attention 实际读取量化缓存。
decode 速度另用逐 token 路径测量。缓存均为 1024 slot、256 token 分页，v_group=64。
FP16 KV 对照调用官方 FlashAttention 分页缓存接口。

## 精度和显存

| 指标 | FP16 KV | KV4 |
|---|---:|---:|
| PPL | 12.315045 | 12.504099 |
| KV 缓存 MiB | 144.000 | 37.195 |
| 执行的 attention 层 | 36 | 36 |

PPL 绝对上升 0.189055，相对上升 **1.535%**。
KV 存储压缩 **3.871×**，减少约 74.2%；这不是整个模型显存压缩比，FP16 权重保持不变。
缓存字节包含 K/V 整数与 scale；平滑参数、模型权重、上下文管理和临时内存另外记录。

首段前 32 个预测位置的 KV4/FP16 logits 相对 L2 为 0.155724，top-1 一致率为 96.875%（31/32）。
该 probe 很小，不应推广为完整数据集的 top-1 指标。
FP16 分块路径与完整 FP16 prefill 的 logits 相对 L2 为 0.002284，top-1 一致率为 100%。
这些是实现对照与量化误差记录，本轮未设定 KV4 的产品精度验收门限。

## 速度

单请求、eager、预热 1 轮、测量 3 轮，decode 每轮 32 步。
时间包含模型、KV 读写及最后位置 lm_head，不含调度、采样和上下文元数据准备。

| 上下文 token | FP16 prefill ms | KV4 prefill ms | FP16 decode ms/token | KV4 decode ms/token | decode 加速比 |
|---:|---:|---:|---:|---:|---:|
| 128 | 27.390 | 29.975 | 22.220 | 25.902 | 0.858× |
| 512 | 66.455 | 65.371 | 23.290 | 25.940 | 0.898× |

本轮没有测到 decode 加速，KV4 的 decode 延迟增加约 11%–17%。
prefill 的 512 token 结果接近，3 轮小规模测量不足以据此认定稳定加速。
FP16 权重读写、当前 Triton attention 的实现开销都会影响整模结果；本轮没有做性能归因剖析。

两种 KV 使用各自实际 prefill 路径：FP16 为官方融合缓存 API，KV4 为量化写入后调用 FlashAttention varlen。
不包含 CUDA graph、W4 权重、投机接受率、多请求吞吐和完整语料评测。

## 复现

```bash
conda activate "$PWD/.conda"
python -m benchmarks.kv4.run --gpu 1
```

服务器原始数据：`results/kv4/2026-10-16/`，包含 JSON、中文报告、逐轮延迟、逐样本 NLL、校准元数据和测试日志。
校准文件：`datasets/kv4/qwen3-8b.pt`。本轮任务进程均已结束。
