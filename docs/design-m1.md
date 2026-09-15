# design-m1: W4A16 量化链路

> M1 设计半页（docs/01 §3 约定）。定稿后动工。

## 目标

HF checkpoint（BF16）→ qslab_w4_v1 打包格式 → engine 以 W4A16 模式 decode-only 推理，正确性对齐 oracle，PPL 劣化 <0.3，kernel ≥1.3× FP16 cuBLAS（decode 区间）。

## 数据流

```
adapters.download（M0 已有模型） → quantizer/calibrate.py（激活统计）
  → quantizer/w4.py（RTN group=128；--algo awq 时先 grid search scaling）
  → quantizer/packfmt.py（qfp uint32 打包 + scale/zero fp16 → tensors.safetensors + config.json + calib.json）
  → qslab/model/loader.py（读 qslab_w4_v1，按 format_version 分派）
  → qslab/model/patched.py（M1 阶段：q_proj 反量化到 fp16 常驻 → cuBLAS；kernels/ 就绪后切换真 kernel）
  → engine 不变
```

## 阶段划分（关键决策）

**M1 分两步走：**
- **M1a（软件链路）**：反量化放 PyTorch（`qfp >> (4*(i%8)) & 0xF` → `(q-zero)*scale`），预先反量化成 fp16 权重常驻显存。目的：先验证"打包格式→加载→数值正确→PPL 达标"整条链路。显存 1.7B 约无压力，8B 时 W4 打包+fp16 常驻共存约 5G+10G=15G，仍可跑（M4 前会切真 kernel 解决）。
- **M1b（kernel）**：w4a16_gemm.cu 替换 q_proj 的 matmul（只换 attention 的 q_proj？不——M1b 把全部 4 个投影都换成 kernel 路径，kv/qk-norm 不动）。验收 1.3× 在 kernel benchmark（test_*.cu 独立测）+ 端到端 tok/s 双口径。

## qslab_w4_v1 格式（docs/04 定稿的落地）

- config.json: `{format_version:1, algo, group_size:128, symmetric:true, quantized_layers:[...], model_config:{...}}`
- tensors.safetensors: 每 Linear 权重三元组 `qfp[u32: O, I/8]`, `scale[fp16: O, I/128]`, `zero[fp16: O, I/128]`（v1 zero 恒 0）
- 不量化：embed_tokens、lm_head（tied）、所有 norm、rotary。量化：q/k/v/o_proj、gate/up/down_proj
- calib.json: `{dataset:"c4", hash:..., n_samples:128, seq_len:2048}`

## 校准器接口

```python
collect_activations(model, calib_ids, layers) -> dict[layer_name, Tensor[O]]  # absmean per-channel
```
用 forward hook 抓每个 Linear 的输入，在线累计 absmean/absmax（不存原始激活）。calib_ids 由 adapters/download.py 的 `build_frozen_calib()` 生成（C4 128×2048，token 存 results/frozen/calib_c4_128x2048.pt）。

## RTN 算法（w4.py 核心 30 行）

```python
scale = amax.abs().amax(dim=-1, keepdim=True) / 7.0     # per-group
q = clamp(round(w / scale.clamp_min(eps)), -8, 7)        # symmetric int4
```
对称量化（zero=0），group=128。AWQ scaling（--algo awq）：s_search ∈ {0,0.05,...,1}，`W·diag(s)` 量化后与 `x/diag(s)` 重算输出 MSE，选最小 s。

## 验收实验矩阵（docs/03）

1. 正确性：W4A16 引擎 vs 反量化 fp16 直接前向，token 一致（同 seed greedy）
2. PPL：qslab_w4 RTN vs AWQ vs FP16（16 docs 口径先与 M0 对齐，冻结前扩 docs）
3. kernel：test_w4a16_gemm.cu，M=4096(K=4096/N=4096 等真实形状) vs torch.matmul fp16 基线
4. 显存：loader 后 process memory 报告

## 不做（M1 范围外）

GPTQ、非对称 zero（格式留位）、权重 shuffle（v2）、KV4（M2）、8B 全量（M4）
