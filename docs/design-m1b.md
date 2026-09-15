# design-m1b: w4a16 GEMM CUDA kernel

> M1b 设计。目标：decode-only batch=1 区间，W4A16 GEMM kernel ≥1.3× FP16 cuBLAS。

## 问题定义

decode 单步的 GEMM 形状：M=1（token 数），N=out_features，K=in_features。
例如 Qwen3-1.7B：q_proj (N=2048,K=2048)，down_proj (N=2048,K=6144)。
M=1 时是 GEMV——显存带宽受限（读权重决定时间），FP16 读 2B/权重，W4 读 0.5B/权重
→ 理论加速比 ≈ 4×（带宽瓶颈下），1.3× 是保守验收线。

## 数据流

```
qfp [N, K/8] u32 ──kernel 内解包──> int4 值
scale [N, K/128] fp16 ──> 反量化 w = (q - zero) * scale   (zero=0: w = q*scale)
x [M, K] fp16 ────────────────────> y = x @ w^T  (fp16 accumulate->fp32->fp16)
```

## 实现策略（简化版，对标 Marlin 的思想但工程量小一个量级）

1. **每 warp 处理一行（N 维）**：K 沿线程分段，每线程负责连续 K 段
2. **解包**：每 u32 有 8 个 int4；warp shuffle 无需——直接按线程 ID 映射到
   (n, k_chunk)，一个 u32 展开 8 个权重，用快速位运算（无查表）
3. **向量化读**：qfp 按 uint32 4 连读（128-bit load），x 按 float4 连读
4. **归约**：warp 内 shuffle reduce（K 段部分和），warp leader 写回 y[n]
5. M>1 小批量（decode 时 M=1；speculative 验证时 M=γ≤8）：每 block 处理
   多行 × 多 token，暂按 M=1 优化，M>1 走同一 kernel（block 沿 N 分布）

## 正确性基准

test_w4a16_gemm.cu：随机生成 fp16 权重 → pack_w4（复用 quantizer 的 CPU 逻辑，
在 host 端做）→ kernel 解包 GEMM vs fp16 直接 GEMM（torch reference）：
- 相对误差 < 1e-2（int4 量化误差主导，不是 kernel 数值 bug 的量级）
- 同时验证 kernel 输出 == "预先反量化成 fp16 再 cuBLAS"（排除解包错误）

## 性能基准

bench 矩阵：真实形状 [(2048,2048), (2048,6144), (6144,2048), (6144,12288)] ×
{kernel, torch.matmul(fp16 权重), torch.matmul(预反量化 fp16)}。
计时 cudaEvent，100 次取中位数。4090 带宽 ~1000GB/s：
- FP16 权重 GEMV：K=6144,N=2048 → 25MB 权重 → 理论 ~25µs
- W4：6.3MB → 理论 ~6.3µs → 4× 加速上限

## 文件

- kernels/csrc/w4a16_gemm.cu（kernel + launcher）
- kernels/csrc/test_w4a16_gemm.cu（独立可执行：正确性+性能）
- kernels/setup.py（torch extension 绑定，供 Python 侧调用）
- kernels/qslab_kernels/ops.py（绑定层）
- scripts/build_kernels.sh（gcc-13 + sm_89 编译）
