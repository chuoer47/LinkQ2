# Marlin 官方接入验证

使用 [官方 Marlin](https://github.com/IST-DASLab/marlin/tree/1f25790bdd49fba53106164a24666dade68d7c90) 的安装脚本、`Layer.pack()`、`Layer.forward()`，源码不改动。
仓库只保留算子所需文件与官方测试，位于 `third_party/marlin`。

## 编译

在仓库根目录执行，所有工具和 Python 依赖放在项目 conda 环境：

```bash
conda activate "$PWD/.conda"
conda install --freeze-installed --override-channels \
  -c https://conda.anaconda.org/nvidia \
  -c https://conda.anaconda.org/conda-forge \
  cuda-nvcc=12.6 cuda-cudart-dev=12.6 cuda-cccl=12.6
python -m benchmarks.marlin.build
```

对应项目 torch 2.7.1+cu126。CUDA 编译器须与 torch CUDA 版本匹配；构建脚本只配置 SM 8.9、项目已有头文件目录，再调用 pip 执行原版 setup.py，不另写编译器或扩展加载流程。
conda 激活脚本设置兼容的 C++ 编译器；CUDA 头文件来自项目工具链及现有 torch 依赖。

## 一键验证

```bash
python -m benchmarks.marlin.run --gpu 1
```

默认读取 `models/Qwen3-8B`、`models/Qwen3-8B-W4-rtn`、`models/Qwen3-8B-W4-awq` 与 `datasets/w4a16/samples.json`。
可用 `--model`、`--checkpoints`、`--samples`、`--out` 调整路径。结果保存在 `results/marlin/<时间>/`。
各阶段依次运行独立进程，失败立即停止；Ctrl+C 会停止当前进程。

验证顺序：

1. 原版 `test.py`。
2. Qwen3 的四种投影形状：N×K 为 4096×4096、1024×4096、12288×4096、4096×12288；M=1/16/128/512。
3. 整模 FP16 基线，再分别加载 RTN/AWQ 包做 fake 与 Marlin PPL 对照。

单算子检查项目封装与官方打包的 B/s 完全相等，使用与官方测试一致的平均绝对相对误差门限 0.001。
另报告相对 L2、最大绝对误差，以及 CUDA graph 热权重延迟和同形状 FP16 加速比。该速度口径不代表模型 decode 收益。

整模使用 WikiText-2 固定 8 段×512 token，共 4088 个预测目标；校准数据为独立 C4 训练片段。
全部 252 个量化层执行，额外做 32 token 上下文和 8 步普通 HF cache decode。
记录 real/fake PPL 差、logits 差异、权重存储字节和释放后显存，real/fake PPL 相对差门限为 1%。
该测试覆盖整个模型，评测语料为小规模固定样本。本轮不运行完整语料或引擎吞吐测试。

## 接入范围

当前量化包可直接选择后端：

```python
from adapters.model_builder import build_quantized_causal_lm

model = build_quantized_causal_lm(
    "models/Qwen3-8B-W4-awq", backend="marlin", device="cuda:0")
```

原版支持 FP16、对称 W4、group size 128（或 -1）；K 需整除 128，N 需整除 256。
本项目现有包使用 group size 128。非对称 AWQ 及 BF16 直接报错，不回退到其他计算路径。
后端在加载时逐层构建临时反量化 FP16 Linear，交给官方 pack 后释放；运行时常驻打包权重、scale、workspace 和可选 input_scale。
AWQ 的输入缩放来自包内明确的算法策略，在线执行 `x / input_scale` 后调用原版 Layer.forward。

模型可执行 `.cpu().cuda()` 移动；Marlin 运算在 CUDA 上执行。GPU 测试包含原浮点权重释放、模型释放和设备移动。
引擎 API 可选 `quant_backend="marlin"`；本轮整模验证使用普通 HF，未做引擎端到端测试。
