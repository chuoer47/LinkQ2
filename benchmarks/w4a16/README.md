# W4A16 小规模 benchmark

本目录验证项目内融合算子的性能、显存收益及 real 与 fake 的精度差异。

## 一键运行

在仓库根目录执行，使用项目专属环境与已有量化产物：

```bash
conda activate "$PWD/.conda"
python -m benchmarks.w4a16.run --gpu 1
```

默认需要 `models/Qwen3-8B`、`artifacts/rtn`、`artifacts/awq`、`datasets/w4a16/samples.json`。
每个产物目录包含 `manifest.json` 与 `tensors.safetensors`。
也可用 `--model`、`--artifacts`、`--algorithms rtn awq awq_affine`、`--samples`、`--out` 指定路径和范围。
各模式顺序使用独立子进程；Ctrl+C 会停止当前子进程。

## 测量范围与口径

- 默认顺序为 FP16、RTN fake/real、AWQ fake/real，不重新校准和量化。
- PPL 固定 WikiText-2 的 8 段×512 token，4088 个预测目标，所有模式共用同一 token 文件；仅用于小样本对照。
- fake 在加载时用 FP32 还原逻辑权重，折叠 input_scale 后转为 FP16 Linear；运行时常驻 FP16 权重。
- real 在加载时将逻辑整数打包成 W4，运行时融合反量化和计算；AWQ 的输入缩放在 forward 中执行。
- prefill：batch=1，128/512 token，普通 HF eager、无 KV cache；预热 2 次、测量 5 次。
- decode：128 token 上下文，再输入固定 32 token，普通 HF FP16 KV cache；预热 2 轮、测量 5 轮。每轮逐步同步，报告轮均值的中位数和原始数据。
- 权重实际存储字节数、PyTorch allocated/reserved、加载和前向峰值、释放后 allocated 分别记录，不能混称为显存。
- 核查每个量化层执行覆盖，报告 real/fake 的 probe 与 decode logits 差异、PPL 差值。

结果保存在 `results/w4a16/<时间>/`：中文报告、各模式 JSON/日志及 autotune 缓存。
第一次编译/autotune 不计入稳态延迟。加速比始终为 FP16 延迟除以 real 延迟，大于 1 才是加速。
本测试不包含调度引擎、量化 KV cache、多请求吞吐或完整语料 PPL。

## 单算子与原 GemLite 比较

```bash
CUDA_VISIBLE_DEVICES=1 python -m benchmarks.w4a16.operator

# 原包源码只作临时对照，无需 pip 安装
CUDA_VISIBLE_DEVICES=1 python -m benchmarks.w4a16.operator \
  --reference /path/to/gemlite-0.4.8 \
  --out results/w4a16/original_comparison.json
```

覆盖三个 Qwen3-8B 投影形状以及一个 affine 形状，M=1/32/128/512。
比较 packed/scales/zeros、输出相对 L2、GPU CUDA graph 延迟和同步主机延迟。
FP16/Fake Linear 使用同形状连续 FP16 权重；这是权重热缓存的单算子实验，不能据此声称整个模型 decode 加速。

## 小阶段测试

```bash
python -m pytest tests/quant/test_quantization.py -m 'not gpu' -q
CUDA_VISIBLE_DEVICES=1 python -m pytest tests/quant/test_quantization.py -m gpu -q
```

GPU 测试验证对称/非对称产物、真实模型替换、CPU/GPU 移动、源权重释放与模型释放后的显存回收。
