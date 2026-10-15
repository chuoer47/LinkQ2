# W4A16 主链路

## 离线量化

`qslab/compression` 从原始 BF16/FP16 checkpoint 加载模型。四个算法各自实现 `WeightQuantizer.quantize()`；pipeline 根据 `requires_calibration` 决定是否用输入 token 重新收集激活。
算法仅产生逻辑整数、scale、zero point 和可选 input_scale，不导入在线后端或 kernel。

## 完整 checkpoint

`qslab.compression.save_checkpoint(artifact, model_path, output)` 按原模型分片逐个写出完整权重包，使用 safetensors 保存张量，使用标准库复制配置和 tokenizer 等原有文件。

- 原始权重名称及分片映射保持一致。
- 量化权重仍使用原始 `<模块>.weight` 名称，INT8 存储逻辑 W4 数值。
- `<原权重名>.scale` 为 FP16；非对称量化增加 `.zero_point`；有输入缩放则增加 `.input_scale`。
- 辅助张量与主权重同分片，加入同一个 `model.safetensors.index.json`，更新实际 `total_size`。单文件源模型继续使用单文件。
- 未量化权重保持源 checkpoint 的数值和 dtype。
- tokenizer、生成配置等原有文件随包保存。

在 `config.json` 增加算法与分组配置：

```json
"quantization_config": {
  "quant_method": "awq_w4a16",
  "bits": 4,
  "group_size": 128,
  "symmetric": true
}
```

另有一个 `quantized_layers.json`，内容仅为模块名称数组，例如：

```json
["model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj"]
```

此数组是替换哪些层的唯一依据。位宽表示逻辑量化精度；紧凑 int4 布局由在线后端加载时生成。包内不绑定具体 kernel。

算法标识只在 `qslab/quantization.py` 的 `QuantizationMethod` 枚举定义：`rtn_w4a16`、`rtn_clip_w4a16`、`awq_w4a16`、`awq_affine_w4a16`。它只含协议名称，不导入离线或在线组件。

## 在线加载

`qslab/runtime/quantization.py` 独立读取完整包的配置与统一索引，不导入离线算法或离线对象。
`load_checkpoint_weights()` 读取 `quant_method`，分发到 `RuntimeQuantization` 的具体实现，再按显式层清单替换 Linear，并加载其余浮点参数。

- `RTNRuntime` / `RTNClipRuntime`：按 K 分组，`W = q × scale`。
- `AWQRuntime`：同样还原权重，forward 显式传入 `x / input_scale`；fake 参考将该变换折叠进权重。
- `AWQAffineRuntime`：`W = (q - zero_point) × scale`，并执行 AWQ 输入缩放。

策略明确要求对应辅助张量；缺少 AWQ 的 input_scale 会报错。层选择与输入缩放均不通过字段是否存在来推断。算法策略解释产物，GemLite 后端负责布局与计算。

HF 使用 `adapters.model_builder.build_quantized_causal_lm()`：通过 Transformers 从配置构造模型、通过 Accelerate 创建空参数并加载张量。只需量化模型目录；`fake=True` 用于构建配对的浮点反量化参考。

项目引擎同样只接收模型目录，检测包内 `quantization_config` 并调用同一加载函数。

`QuantLinear` 定义后端接口；`GemLiteLinear` 负责打包，所有常驻张量注册为 buffer。forward 按需执行输入缩放，调用融合反量化与矩阵计算算子。
kernel 来自 GemLite 0.4.8，计算源码保持原样。M=1 使用 revsplitK GEMV，2≤M≤64 使用 splitK GEMM，M>64 使用 GEMM。

## 验证

`tests/quant` 覆盖量化算法、完整分片、浮点权重保留、删除测试原目录后的独立加载，以及设备移动和释放。
`benchmarks.w4a16.validate_checkpoint` 验证实际完整包，并配对比较 real/fake 的小样本 PPL、probe 与短 decode。
`benchmarks.w4a16.run` 测量 HF prefill/decode 性能，单算子对照使用 `benchmarks.w4a16.operator`。
