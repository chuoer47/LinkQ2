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
- [x] M1-S6 W4 链路 oracle 对齐（引擎 W4 路径 vs 直接反量化：2 prompts×32tok ALL_MATCH）（2026-09-15）
- [x] M1-S7 throughput：FP16 复测 42.61 tok/s（与 M0 一致）；W4 反量化路径=FP16（无 kernel，正常）（2026-09-15）
- [x] M1-S8 notes/M1a-软件链路.md（2026-09-15）
- [ ] M1-S9（M1b）kernel：**进行中**——w4a16_gemm.cu 第一版编译跑通（sm_89，gcc-13 链路 4 个 leetcuda 坑全踩通）。首测：大形状 (6144,12288) 达 **3.59×** vs fp16 GEMV；小形状 (2048,2048) 仅 0.90×（warp 并行度不足）。下一步：小形状优化（block 内多 warp / 向量化读）
- [x] M1-S9（M1b）kernel：w4a16_gemm v5 完成。kernel 级 1.45/1.64/1.77×（主要 decode 形状达标），6144×12288 1.14× 未达；数值 rel err 7e-8；e2e 0.95×（结构性打平，8B 时收益显现）；notes/M1b-kernel.md（2026-09-15）
- [ ] M1-S10 决策点：PPL +1.24 vs 0.3 线 gap——**等用户裁决**（选项见下）

### M1 断点

**M1 全部子任务完成，停在里程碑验收点。** 验收数据：PPL 27.72 (+1.24 vs FP16 26.48) / kernel 1.45~1.77×（主形状）/ token 级对齐全 PASS / 打包 5.3× 压缩 / e2e 0.95×。
S10 选项：(a) 扩大校准集重跑争取 PPL<1；(b) 接受 +1.24 记录分析（推荐，8B M4 时复测）；(c) 调验收线。

## 已完成里程碑

（暂无）

## 外部阻塞记录

（无）

## 环境速记

- 服务器：4090_public（4×4090D sm_89，无系统 CUDA），项目根 `/home/<user>/<workdir>/qserve-lab/`
- env：`qslab`（S1 建）；参照 env：`vllm`（对照组基线）、`leetcuda`（CUDA 工具链方案参考）
- 模型：Qwen3-1.7B/0.6B 在 `/home/<user>/<workdir>/LinkQuant/models/`（M0 先软链用，不复制）；Qwen3-8B 待 M0 验收后下载
- GPU 使用纪律：用最空卡，`CUDA_VISIBLE_DEVICES` 锁定
