# W4A16 主链路

## 离线量化

`qslab/compression` 从原始 BF16/FP16 checkpoint 加载模型。四个算法各自实现 `WeightQuantizer.quantize()`；pipeline 根据 `requires_calibration` 决定是否用输入 token 重新收集激活。
算法仅产生逻辑整数、scale、zero point 和可选 input_scale，不导入在线后端或 kernel。

## 完整 checkpoint

`qslab.artifacts.save_checkpoint(artifact, model_path, output)` 按原模型分片逐个写出完整权重包，使用 safetensors 保存张量，使用标准库复制配置和 tokenizer 等原有文件。

- 原始权重名称及分片映射保持一致。
- 量化权重仍使用原始 `<模块>.weight` 名称，INT8 存储逻辑 W4 数值。
- `<原权重名>.scale` 为 FP16；非对称量化增加 `.zero_point`；有输入缩放则增加 `.input_scale`。
- 辅助张量与主权重同分片，加入同一个 `model.safetensors.index.json`，更新实际 `total_size`。单文件源模型继续使用单文件。
- 未量化权重保持源 checkpoint 的数值和 dtype。
- tokenizer、生成配置等原有文件随包保存。

仅在 `config.json` 增加：

```json
"quantization_config": {"bits": 4, "group_size": 128, "symmetric": true}
```

位宽表示逻辑量化精度；紧凑 int4 布局由在线后端加载时生成。包内不绑定具体 kernel。

## 在线加载

`qslab/runtime/quantization.py` 独立读取完整包的配置与统一索引，不导入离线算法或离线对象。
`load_checkpoint_weights()` 替换量化 Linear，并加载其余浮点参数。

HF 使用 `adapters.model_builder.build_quantized_causal_lm()`：通过 Transformers 从配置构造模型、通过 Accelerate 创建空参数并加载张量。只需量化模型目录；`fake=True` 用于构建配对的浮点反量化参考。

项目引擎同样只接收模型目录，检测包内 `quantization_config` 并调用同一加载函数。

`QuantLinear` 定义后端接口；`GemLiteLinear` 负责打包，所有常驻张量注册为 buffer。forward 按需执行输入缩放，调用融合反量化与矩阵计算算子。
kernel 来自 GemLite 0.4.8，计算源码保持原样。M=1 使用 revsplitK GEMV，2≤M≤64 使用 splitK GEMM，M>64 使用 GEMM。

## 验证

`tests/quant` 覆盖量化算法、完整分片、浮点权重保留、删除测试原目录后的独立加载，以及设备移动和释放。
`benchmarks.w4a16.validate_checkpoint` 验证实际完整包，并配对比较 real/fake 的小样本 PPL、probe 与短 decode。
`benchmarks.w4a16.run` 测量 HF prefill/decode 性能，单算子对照使用 `benchmarks.w4a16.operator`。
