# PROGRESS.md — qserve-lab 唯一状态源

> Loop 每轮先读本文件。更新规则：完成子任务即更新状态+断点，commit 格式 `[M0-xxx] 摘要`。
> 设计依据：docs/00~04（冲突时以 docs/ 为准）。

## 项目主线（速览）

Qwen3-8B + W4A16（自写 kernel）+ KV4 + draft-model 投机推理，decode-only 单请求，4090 单卡。
里程碑：M0 FP16 基线 → M1 W4A16 → M2 KV4 → M3 投机推理 → M4 组合+128K demo。

## 当前里程碑：**M1 — W4A16 量化链路**（M0 已完成：token 级对齐 oracle，42.08 tok/s，PPL 26.48）

### M1 子任务清单（design 见 docs/design-m1.md）

- [x] M1-S1 quantizer 骨架：packfmt(u32 nibble)+w4(RTN/AWQ)+calibrate(hook)，roundtrip PASS（2026-09-14）
- [x] M1-S2 RTN 全模型量化：196 Linear 层 3.8G→715M 打包（2026-09-14）
- [x] M1-S3 校准集冻结 results/frozen/calib_c4_128x2048.pt + AWQ 量化完成（2026-09-15）
- [x] M1-S4 PPL 对比：FP16 26.48 / RTN 33.97(+7.49) / AWQ 32.27(+5.79)——**劣化远超 0.3 验收线，排查中**（2026-09-15）
- [x] M1-S5 **排查 PPL 劣化**：3 轮算法实验——修 where 广播 bug；s 搜索重写（全局选优）；per-group s（+1.81 更差）；全 clip 范围（+2.17 更差）。**最优=全局 s + clip 0.4：PPL +1.24**（awq2 配方）。已确认链路无罪，算法迭代到收益递减点（2026-09-15）
- [ ] M1-S6 W4 链路 oracle 对齐（W4 反量化前向 vs 引擎 W4 路径 token 一致）
- [ ] M1-S7 throughput W4 vs FP16 对比（M1a 软件路径）
- [ ] M1-S8 notes/M1-*.md 实验心得（含 RTN+7.49→AWQ+1.24 的完整调优叙事）
- [ ] M1-S9（M1b）kernel：w4a16_gemm.cu + test bench，≥1.3× 验收
- [ ] M1-S10 决策点：PPL +1.24 vs 0.3 线 gap——分析误差传导（校准质量/未量化 embed/QK-norm 路径），或与用户商议调整验收线/增加 GPTQ

### M1 断点

算法收敛：awq2 配方（全局 s + clip 0.4 + 激活加权 clip）= PPL 27.72 (+1.24)。三个对照实验（per-group s / 全 clip / rtn）均劣于它。下步：M1-S6 oracle 对齐 → M1-S7 吞吐 → kernel。+1.24 vs 0.3 的 gap 处理见 S10 决策点。

## 已完成里程碑

（暂无）

## 外部阻塞记录

（无）

## 环境速记

- 服务器：4090_public（4×4090D sm_89，无系统 CUDA），项目根 `/home/<user>/<workdir>/qserve-lab/`
- env：`qslab`（S1 建）；参照 env：`vllm`（对照组基线）、`leetcuda`（CUDA 工具链方案参考）
- 模型：Qwen3-1.7B/0.6B 在 `/home/<user>/<workdir>/LinkQuant/models/`（M0 先软链用，不复制）；Qwen3-8B 待 M0 验收后下载
- GPU 使用纪律：用最空卡，`CUDA_VISIBLE_DEVICES` 锁定
