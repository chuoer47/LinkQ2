# M1b — W4A16 CUDA kernel：实验心得

> 2026-09-15 · RTX 4090 D (GPU3) · w4a16_gemm v5（共享 LUT 版）

## Kernel 性能（L2 冲刷后，真实 DRAM 测量）

| 形状 (N×K) | w4 kernel | fp16 GEMV 基线 | 加速比 |
|---|---|---|---|
| 2048×2048 | 6.4 µs | 10.5 µs | **1.64×** ✅ |
| 2048×6144 | 13.3 µs | 23.6 µs | **1.77×** ✅ |
| 6144×2048 | 11.3 µs | 16.4 µs | **1.45×** ✅ |
| 6144×12288 | 55.3 µs | 63.3 µs | 1.14× ⚠️ |

数值：kernel vs 反量化 matmul rel err **7e-8**；token 级 e2e 对齐 PASS（2 prompts × 32 tok）。

## 版本演进（3 轮优化都无效或负收益，最终版=正确合并读）

- v1：warp/row，逐 word 读取。基线：0.90×（cache 未冲刷时假象 3.59×）
- v2：4 warp 分 K + shared 归约——**更慢（0.42×）**：chunk 交错破坏了 128B 合并读
- v3：warp/row + uint4 向量化（16B/lane）——与 v1 持平（v1 的 32×4B 已是完美 coalescing）
- v4：block 覆盖 4 行摊薄启动开销——小形状受益
- v5：共享内存 LUT（16 项/group）——无增益，编译器已做同等优化

**核心教训：coalescing 敏感度 > 一切。** v2 的教训是 warp 间数据布局交错会把 16B 向量读拆散；而"每 lane 连续 16B、跨 lane 连续递增"才是正解。

## 测量方法论（比 kernel 本身更有传播价值的发现）

1. **4090 的 72MB L2 会完整缓存 ≤25MB 的权重矩阵**：不冲刷 L2 的 GEMV benchmark 虚高 2~4×（FP16 基线一度测出 9.2TB/s 的物理不可能值）。修复：每迭代前 memset 96MB 缓冲区驱赶 L2。
2. **CPU dispatch 开销的量级**：`x @ w.T`（cuBLAS 路径）单次 submit 1.1ms CPU 时间，自定义扩展 0.2ms——但真实引擎里 async 队列摊薄了它，isolated microbench 的结论不能直接外推到 e2e。
3. **e2e vs kernel 级的鸿沟**：1.7B 单步 21.5ms 里 GEMV 总时间只有 ~2-3ms（embed/lm_head 不量化 + attention/归一化占大头），kernel 就算 4× 也只省 ~1ms，而 196 次 Python wrapper dispatch 吃掉等量——**e2e 0.95× 打平是结构性结果，不是 kernel 失败**。8B 模型（M4）权重 4 倍、GEMV 占比 3 倍，同样 kernel 的 e2e 收益会显现。

## nibble 解码 bug（0.11 假误差的教训）

初版 test 的 `(q-8)` 解码与 pack 的 `q&0xF` 编码不匹配（正确解码是 `(q^8)-8`），但 test 的对照是"未量化 fp16 GEMV"，0.11 的 rel_err 恰好落在 int4 量化误差的量级范围内——**被掩盖了整整四轮**。直到 API 扩展版与"反量化 matmul"对拍才暴露（rel err 10.6 → 修后 7e-8）。

教训：**正确性测试必须有一个"零容差"对拍物**（数值上应完全一致的实现），只和"含已知误差的参照物"比是测不出实现 bug 的。

## 验收对照（docs/03 M1 线）

- kernel 吞吐 ≥1.3× FP16 cuBLAS：**主要 decode 形状达标**（1.45/1.64/1.77×），最大 MLP 形状 1.14× 未达（ALU 解包开销在该形状占比最高；Marlin 级优化需寄存器重排+mma 管线，超出本阶段范围）
- PPL <0.3：**+1.24 未达**（S10 决策点，见 M1a 笔记）
- 与 FP16 的 e2e 打平（0.95×），正确性全线 PASS
