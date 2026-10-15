# LinkQ2

面向 RTX 4090 系列的 W4A16、KV 量化、简易推理引擎与投机推理实验项目。

## 环境

所有 Python 依赖放在项目专属 conda 环境中：

```bash
conda env create --prefix ./.conda -f environment.yml
conda activate "$PWD/.conda"
python -m pip install --no-deps -e .
```

当前验证版本为 Python 3.10、torch 2.7.1/CUDA 12.6、Triton 3.3.1。
W4A16 使用项目内 Triton 源码，首次运行会编译和 autotune，无需安装 GemLite。

## W4A16 主链路

离线：`compression → artifacts`，只负责量化与校准。
在线：`runtime.quantization → QuantLinear → GemLiteLinear → kernels/gemlite`，负责加载、后端打包、模型替换与执行。

产物只有 `manifest.json` 与 `tensors.safetensors`；manifest 保存版本、group size 和是否对称，张量名对应模型权重名。模型结构来自原始 HF checkpoint。

```bash
# 离线 RTN；AWQ 增加 --algorithm awq --calibration <token-id-cache.pt>
python -m qslab.compression --model models/Qwen3-8B --algorithm rtn --out artifacts/rtn

# 已有 RTN/AWQ 产物的 HF 小规模对照
python -m benchmarks.w4a16.run --gpu 1

# 打包算子与原生 FP16/Fake Linear 对照
CUDA_VISIBLE_DEVICES=1 python -m benchmarks.w4a16.operator
```

详细测量口径、原包对照命令见 [benchmark README](benchmarks/w4a16/README.md)，架构边界见 [主链路说明](docs/PROJECT_MAINLINE.md)。

推理引擎通过 `LLM(..., quantization="artifacts/rtn", quant_backend="gemlite")` 使用同一正式后端。引擎的 FlashAttention 依赖需另行按 torch/CUDA 版本安装；HF benchmark 不依赖它。
