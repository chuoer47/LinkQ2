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
- [x] M1-S10 决策点：**用户裁决 (b)——接受 PPL +1.24 记录分析，M4 的 8B 复测**（2026-09-15）

### M1 状态：**完成，用户验收通过（2026-09-15，S10 选 b）**

## 当前里程碑：**M2 — KV cache 4bit 量化**

### M2 子任务清单

- [ ] M2-S0 design-m2：KV4 方案定稿（K per-channel / V per-token 不对称，前 N 层 FP16，读时反量化先行）
- [ ] M2-S1 kv4_plan.py：离线测每层 KV 离群程度 → 生成保 FP16 层清单，写进打包 config
- [ ] M2-S2 kv_cache.py：KV8Cache / KV4Cache 实现（量化写入 + 读时反量化接口）
- [ ] M2-S3 PatchedQwen3Attention 接量化 cache（per-layer 可选精度）
- [ ] M2-S4 正确性：KV4 引擎 vs FP16 KV 引擎，PPL 对比（验收线 <0.5）+ 显存对比（≥3.5×）
- [ ] M2-S5 NIAH 32K：召回率对比（验收线降幅 <5%）
- [ ] M2-S6 notes/M2-kv4.md

### M2 状态：**完成，用户验收通过（2026-09-16，边缘项记为 M4 复测项）**

## 当前里程碑：**M3 — draft-model 投机推理**

### M3 子任务清单

- [ ] M3-S0 design-m3：拒绝采样数学 + draft/target 调度设计
- [ ] M3-S1 sampler.py：top-k/top-p/temperature 采样 + 拒绝采样核心（含无损性证明的单测）
- [ ] M3-S2 spec/draft.py：Qwen3-0.6B draft 包装（同 tokenizer，engine 复用）
- [ ] M3-S3 spec/verify.py：γ-token 批量验证 + 回滚逻辑
- [ ] M3-S4 engine 集成：draft/target 交替编排循环
- [ ] M3-S5 正确性：无损性验证（固定 seed 下与 target-only 采样一致率 >99%）
- [ ] M3-S6 bench_spec.py：acceptance rate + γ 扫描（1~6）+ 端到端加速比（验收线：ar≥2 时 ≥1.5×）
- [ ] M3-S7 notes/M3-spec.md

### M3 状态：**完成，待用户验收**

验收数据：无损性 **100% token 一致**（3 prompts×64tok，线 >99%）✅；AR=4.96@γ4 但 e2e **0.74×**（线：AR≥2 时 ≥1.5×）❌——负加速根因：Python 调度开销主导小模型场景，γ 扫描与四点分析在 notes/M3-spec.md。M4 的 8B target 是转正的实验点。
注：docs/03 的加速线在 1.7B 上结构性达不到（模型小 + Python 引擎 dispatch 重），如实记录，8B 复测。

### M3 断点

（无——M3 完成，等用户确认进 M4）

## 已完成里程碑

（暂无）

## 外部阻塞记录

（无）

## 环境速记

- 服务器：4090_public（4×4090D sm_89，无系统 CUDA），项目根 `/home/<user>/<workdir>/qserve-lab/`
- env：`qslab`（S1 建）；参照 env：`vllm`（对照组基线）、`leetcuda`（CUDA 工具链方案参考）
- 模型：Qwen3-1.7B/0.6B 在 `/home/<user>/<workdir>/LinkQuant/models/`（M0 先软链用，不复制）；Qwen3-8B 待 M0 验收后下载
- GPU 使用纪律：用最空卡，`CUDA_VISIBLE_DEVICES` 锁定
