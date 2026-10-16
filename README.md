# LinkQ2

面向 RTX 4090 系列的 W4A16、KV 量化、简易推理引擎与投机推理实验项目。

## 环境

```bash
conda env create --prefix ./.conda -f environment.yml
conda activate "$PWD/.conda"
python -m pip install --no-deps -e .
```

所有 Python 依赖使用项目专属 conda 环境。当前验证版本为 Python 3.10、torch 2.7.1/CUDA 12.6、Triton 3.3.1。W4A16 使用项目内 Triton 源码。

## 从原始模型生成完整 checkpoint

```bash
CUDA_VISIBLE_DEVICES=1 python -m qslab.compression \
  --model models/Qwen3-8B --algorithm rtn_w4a16 --out models/Qwen3-8B-W4-rtn

CUDA_VISIBLE_DEVICES=1 python -m qslab.compression \
  --model models/Qwen3-8B --algorithm awq_w4a16 \
  --calibration datasets/w4a16/checkpoint_calibration.pt \
  --out models/Qwen3-8B-W4-awq
```

源 checkpoint 按其原始 dtype 加载。AWQ 校准文件包含 `token_ids`，建议使用与评测集分离的少量训练片段。
输出目录必须尚不存在。

输出沿用原模型文件、权重名称与分片归属。量化投影的原 `.weight` 张量替换为 INT8 存储的逻辑 W4 整数；scale、可选 zero_point/input_scale 存在同一分片，并登记到同一个权重索引。其余浮点权重、tokenizer、生成配置等随包保存。`config.json` 的 `quantization_config.quant_method` 标明算法，`quantized_layers.json` 明确列出需要替换的模块。

## 仅从新目录加载

```python
from adapters.model_builder import build_quantized_causal_lm
from transformers import AutoTokenizer

path = "models/Qwen3-8B-W4-rtn"
model = build_quantized_causal_lm(path, device="cuda:0")
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
```

模型结构从包内配置构造，再加载包内全部权重。GemLite 后端加载时打包为 int4，forward 调用融合算子。
Marlin 后端使用官方 `Layer.pack()` 和 `Layer.forward()`，可传入 `backend="marlin"`；当前实际形状数值验证未通过，编译与状态见 [Marlin benchmark](benchmarks/marlin/README.md)。
引擎使用 `LLM(path, quant_backend="gemlite")`，自动根据包内配置选择量化加载。引擎的 FlashAttention 依赖需按 torch/CUDA 版本另行安装。
引擎默认使用 FP16 KV、eager 执行。传入 `kv_cache_dtype="kv4", smooth_kv=<校准文件>` 启用 KV4，与权重量化独立；文件中的 lambda 和静态 K scale 须为有限正数。`v_group` 同时决定 KV4 池布局和 Attention 的读写粒度。FP16 KV 使用 nano-vLLM 的存储算子与官方 FlashAttention 分页接口，页面大小须为 256 的倍数；传入 `enforce_eager=False` 或 CLI 的 `--cuda-graph` 开启 decode CUDA Graph。

```python
from qslab.api import LLM, SamplingParams

with LLM("models/Qwen3-8B", max_num_seqs=4) as llm:
    result = llm.generate("The capital of France is", SamplingParams(max_tokens=16))
```

输入必须非空且短于 `max_model_len`，输出预算按剩余上下文截断；单条请求无法容纳于 KV 池时立即拒绝。请求完成、取消或执行异常后归还页面，`llm.close()` 释放模型与池；前缀缓存可保留零引用页面内容用于复用。HTTP 断连会取消请求，执行线程失败会通知等待中的客户端。

`enable_chunked_prefill=True, max_num_batched_tokens=64` 开启 decode 优先的混合分块预填充；纯 decode 保持 CUDA Graph，含 prefill 的批次使用 varlen。CLI/HTTP 使用 `--chunked-prefill --max-num-batched-tokens 64`。该策略来自 nano-vLLM PR #276，当前不与投机推理组合，来源和适配见 [上游说明](qslab/runtime/UPSTREAM.md)。

## 验证和 benchmark

```bash
python -m benchmarks.w4a16.validate_checkpoint --gpu 1
python -m benchmarks.w4a16.run --gpu 1
```

第一条验证完整包、tokenizer、独立加载、8×512 token 的 real/fake PPL 与短 decode；第二条另测 HF prefill/decode 性能。
KV4 与 FP16 KV 的独立对照使用 `python -m benchmarks.kv4.run --gpu 1`，范围与计时口径见 [KV4 README](benchmarks/kv4/README.md)。
引擎第一版的四种权重/KV 组合、生命周期和端到端延迟使用 `python -m benchmarks.engine.run --gpu 1`，见 [引擎 README](benchmarks/engine/README.md)。
CUDA Graph 配对对照使用 `python -m benchmarks.cuda_graph.run --gpu 1`，见 [Graph README](benchmarks/cuda_graph/README.md)；W4 的严格数值一致性限制见其结果记录。
混合 Chunked Prefill 三请求对照使用 `python -m benchmarks.chunked_prefill.run --gpu 1`，见 [分块预填充 README](benchmarks/chunked_prefill/README.md)。
详细口径见 [benchmark README](benchmarks/w4a16/README.md)，架构见 [主链路说明](docs/PROJECT_MAINLINE.md)。
