# PROGRESS.md — qserve-lab 唯一状态源

> Loop 每轮先读本文件。更新规则：完成子任务即更新状态+断点，commit 格式 `[M0-xxx] 摘要`。
> 设计依据：docs/00~04（冲突时以 docs/ 为准）。

## 项目主线（速览）

Qwen3-8B + W4A16（自写 kernel）+ KV4 + draft-model 投机推理，decode-only 单请求，4090 单卡。
里程碑：M0 FP16 基线 → M1 W4A16 → M2 KV4 → M3 投机推理 → M4 组合+128K demo。

## 当前里程碑：**M0 — FP16 decode-only 引擎**

### M0 子任务清单

- [x] S0 脚手架：目录骨架 + docs 同步 + git init + 本文件（2026-09-14）
- [x] S1 环境：conda 建 `qslab` env（py3.11 / torch 2.5.1+cu124 / transformers 4.57.6 / nvcc 12.4 / gcc-13），按 docs/02 配置单，装完跑验证命令（2026-09-14 全绿：nvcc 12.4 + gcc 13.4.0 + cuda_available True cap(8,9)。坑记录：清华 nvidia channel 404 → 用官方 URL；pip 大包须 nohup 后台+轮询）
- [ ] S2 基础包骨架：pyproject + qslab/config.py + adapters/tokenizer.py（能 tokenize 一个字符串）
- [ ] S3 模型加载：qslab/model/loader.py（safetensors 读取 + 权重映射），transformers 建模（Qwen3ForCausalLM）加载 Qwen3-1.7B
- [ ] S4 patched.py：Qwen3Attention 替换子类 + KV cache 容器接入（fp16 实现先行）
- [ ] S5 engine.py：decode-only 主循环（input 512 → decode 128）
- [ ] S6 正确性验证：与 transformers 原生 generate 对比，固定 seed token 序列一致（oracle 对齐）
- [ ] S7 基线测量：bench_throughput.py 跑 1.7B FP16 tokens/s + PPL（WikiText-2），结果 json 进 results/，对照验收线（PPL 与参考实现误差 <0.1）
- [ ] S8 notes/M0-基线.md：实验心得

### M0 断点

（无——S2 基础包骨架进行中）

## 已完成里程碑

（暂无）

## 外部阻塞记录

（无）

## 环境速记

- 服务器：4090_public（4×4090D sm_89，无系统 CUDA），项目根 `/home/<user>/<workdir>/qserve-lab/`
- env：`qslab`（S1 建）；参照 env：`vllm`（对照组基线）、`leetcuda`（CUDA 工具链方案参考）
- 模型：Qwen3-1.7B/0.6B 在 `/home/<user>/<workdir>/LinkQuant/models/`（M0 先软链用，不复制）；Qwen3-8B 待 M0 验收后下载
- GPU 使用纪律：用最空卡，`CUDA_VISIBLE_DEVICES` 锁定
