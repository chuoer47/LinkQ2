# 项目架构与主链路

项目围绕离线量化、W4A16/KV4 在线执行、自研推理引擎与投机推理组织。用户入口为 `qslab.api.LLM`、`python -m qslab.api.cli generate` 和 `python -m qslab.api.server`。三个入口使用同一套采样参数；CLI 与 HTTP 共用模型、量化及投机选项。

## 目录职责

| 位置 | 职责 |
|---|---|
| `qslab/api` | Python API、命令行、HTTP；共用选项在 `options.py` |
| `qslab/compression` | 离线算法、激活校准和完整 checkpoint 生成 |
| `qslab/runtime/engine` | 请求编排、调度预算、验证后提交和生命周期 |
| `qslab/runtime/execute` | 目标模型执行、输入组装、通用 decode/verify Graph、采样 |
| `qslab/runtime/speculation` | proposer 创建、自回归草稿和 DFlash 的模型/KV/Graph 适配 |
| `qslab/runtime/model` | Qwen3、Attention、RoPE、基础层与浮点权重加载 |
| `qslab/runtime/state` | Sequence、KV 页分配、有效长度和前缀索引 |
| `qslab/runtime/backends` | QuantLinear 接口和 GemLite 布局转换/执行适配 |
| `qslab/runtime/quantization.py` | 完整量化 checkpoint 在线读取与模块替换 |
| `qslab/runtime/quantization_strategies.py` | 按量化算法解释整数、scale 和输入变换 |
| `qslab/speculation` | 不依赖请求、KV 或 Graph 的纯提议与接受算法 |
| `qslab/kernels` | GemLite、KV4 和 FP16 KV 算子 |
| `adapters/model_builder.py` | 离线及 benchmark 使用的 Transformers 模型构造 |
| `benchmarks` | 按功能组织的一键实验、中文说明与结果 |

## 推理引擎

`LLM → LLMEngine → Scheduler → ModelRunner → Qwen3 / Attention → kernels`。

引擎负责请求进出、提议与目标验证的顺序；Scheduler 只管理预算、抢占、已提交 token 和 KV 有效长度。BlockManager 维护页面所有权。ModelRunner 负责加载、KV 池、批输入组装和执行，不实现接受算法。

`runtime/execute/cuda_graph.py` 的 `CUDAGraphs` 管理目标 Graph。M=1 decode 和固定 M=k+1 verify 共享捕获、缓冲区更新与重放代码，以不同 GraphFamily 保存固定地址的缓冲区。含 prefill 的混合批次使用实际长度的 varlen 路径。DFlash 专用的 BF16 StaticCache Graph 位于投机目录，使用官方草稿 forward。

`runtime/sampling_params.py` 是唯一采样配置；默认 greedy、最多 128 token。`qslab.api` 直接导出这个类型。设备、Graph、模型/KV 容量等执行配置集中在 `runtime/config.py`。

## KV4

`scripts/build_smooth_kv.py` 离线生成 SmoothAttention 的 lambda 和静态 K scale。运行时由 ModelRunner 加载校准，Attention 执行 Q/K 逆缩放。K 使用静态逐通道 scale，V 使用动态逐 token、逐组 scale；缓存的存储、物化与分页 attention 由 `qslab/kernels/kv4` 提供。

FP16 KV 与 KV4 的类型独立于权重量化。分块 prefill 读取已存前缀并处理新块；纯 decode/verify 读取分页池，Graph 填充行不覆盖有效 KV。目标缓存与草稿缓存各自维护。

KV4 计算 tile 与物理页大小独立，低并行度 decode 使用 split-KV；多 query 验证按 KV head 复用解包，使用 FP16 Tensor Core 输入及 FP32 softmax/累加。前缀物化为融合 Triton kernel。性能与精度边界见 [专项 benchmark](../benchmarks/kernels/kv4/RESULTS.md)。

## 投机推理与 DFlash 导航

从 [runtime/speculation/README.md](../qslab/runtime/speculation/README.md) 开始阅读。

`factory.prepare_proposer` 读取草稿配置，确定目标特征与验证块长；`factory.create_proposer` 创建 n-gram、lookahead、自回归草稿或 DFlash。`engine/llm_engine.py` 编排提议、目标验证、接受和提交；`qslab/speculation/acceptance.py` 只计算接受结果。

DFlash 的请求状态与提议位于 `runtime/speculation/dflash.py`，专用 StaticCache 和 Graph 位于同目录 `dflash_graph.py`。官方模型主体由 checkpoint 的 auto_map 指向 `models/Qwen3-8B-DFlash-b16/dflash.py`。项目直接调用官方 forward；目标 embedding、lm_head 和指定层特征供草稿复用。

每轮只将已确认且已计算的目标特征送入草稿；结束、取消、异常、抢占时随请求释放状态。目标验证使用固定 M=16 Graph，混合 prefill/verify 继续使用实际查询长度。非 thinking、greedy 为当前 DFlash 使用范围；草稿权重与 KV 均保持 BF16。

## 依赖方向

离线量化与在线 runtime 不互相导入，通过完整 checkpoint 交换逻辑整数与辅助张量。`qslab/quantization.py` 仅维护算法名称枚举。量化布局转换属于在线后端；算法不绑定 kernel。

runtime 使用 Transformers 读取配置、分词器及官方 DFlash 草稿，因此 Transformers 是正式依赖。HTTP 依赖为可选的 aiohttp。所有 Python 依赖安装于项目 `.conda`。

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

在线策略在 `runtime/quantization_strategies.py` 中统一维护。缺少 AWQ 的 input_scale 会报错；层选择与输入缩放均不通过字段是否存在来推断。算法策略解释产物，GemLite 后端负责布局与计算。

HF 使用 `adapters.model_builder.build_quantized_causal_lm()`：通过 Transformers 从配置构造模型、通过 Accelerate 创建空参数并加载张量。只需量化模型目录；`fake=True` 用于构建配对的浮点反量化参考。

项目引擎同样只接收模型目录，检测包内 `quantization_config` 并调用同一加载函数。

`QuantLinear` 定义后端接口；`GemLiteLinear` 负责打包，所有常驻张量注册为 buffer。forward 按需执行输入缩放，调用融合反量化与矩阵计算算子。
kernel 来自 GemLite 0.4.8，计算源码保持原样。M=1 使用 revsplitK GEMV，2≤M≤64 使用 splitK GEMM，M>64 使用 GEMM。

AWQ input_scale 在算子外应用，原始离线包不绑定后端。布局、计算与验证见 [Kernel 专题](kernels/README.md)。

## 验证

正确性测试按四个模块组织，运行方法和覆盖范围见 [测试说明](../tests/README.md)：离线算法与完整包在 `tests/compression`，算子数值在 `tests/kernels`，调度、KV 页和执行状态在 `tests/runtime`，接受概率、草稿缓存与验证提交在 `tests/speculation`。

测试使用小张量及临时生成的小模型，检查数值误差和状态不变量；不依赖固定自然语言输出，也不设吞吐或加速比门槛。真实模型 PPL、耗时、吞吐、显存及场景收益由 [benchmark](../benchmarks/README.md) 单独测量和记录。

`benchmarks.compression.weights.run` 比较完整权重包的 real / fake PPL 和存储量；`benchmarks.runtime.engine.run` 测量完整请求延迟；单算子计时使用 `benchmarks.kernels.w4a16.run`。
