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
- [ ] M1-S5 **排查 PPL 劣化**：加 clip search（AWQ MSE clipping）；若仍差逐层定位
- [ ] M1-S6 W4 链路 oracle 对齐（W4 反量化前向 vs 引擎 W4 路径 token 一致）
- [ ] M1-S7 throughput W4 vs FP16 对比（M1a 软件路径，记录数据非验收）
- [ ] M1-S8 notes/M1-*.md 实验心得
- [ ] M1-S9（M1b）kernel：w4a16_gemm.cu + test bench，≥1.3× 验收

### M1 断点

排查 PPL：RTN +7.49 过大。pack/unpack 已验证无罪（与直接模拟 0 差异）。怀疑对称量化缺 clip search（AWQ 另一半收益）或 hook 统计问题。下步：w4.py 加 clip search 后重测 RTN。

## 已完成里程碑

（暂无）

## 外部阻塞记录

（无）

## 环境速记

- 服务器：4090_public（4×4090D sm_89，无系统 CUDA），项目根 `/home/<user>/<workdir>/qserve-lab/`
- env：`qslab`（S1 建）；参照 env：`vllm`（对照组基线）、`leetcuda`（CUDA 工具链方案参考）
- 模型：Qwen3-1.7B/0.6B 在 `/home/<user>/<workdir>/LinkQuant/models/`（M0 先软链用，不复制）；Qwen3-8B 待 M0 验收后下载
- GPU 使用纪律：用最空卡，`CUDA_VISIBLE_DEVICES` 锁定
