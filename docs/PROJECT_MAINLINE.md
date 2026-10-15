# W4A16 主链路

## 离线

`qslab/compression` 使用原模型和可选校准 token 产生量化结果。RTN、RTN clip、AWQ 对称、AWQ affine 不导入执行后端或 kernel。

`qslab/artifacts` 负责离线产物读写：

```json
{"version": 1, "group_size": 128, "symmetric": true}
```

每个产物统一 group size 与对称性。safetensors 保存：

- `<模块>.weight.values`：形状 `[N,K]`，int8 逻辑值；对称为 −8…7，非对称为 0…15。
- `<模块>.weight.scale`：`[N,K/group_size]`，FP16。
- `<模块>.weight.zero_point`：仅非对称量化需要。
- `transforms.input_scale.<模块>.weight`：可选的输入缩放 `[K]`。

这里的 int8 是便于后端处理的离线逻辑表示；GPU 常驻权重在加载时打包为 int4。模型结构从原始 HF checkpoint 读取，算法及实验来源记录在实验结果中，不扩展运行时协议。

## 在线

`qslab/runtime/quantization.py` 用 safetensors 加载逻辑张量，按名称替换模型中的 Linear，不导入离线算法或离线对象。

`qslab/models/quant_linear.py` 只有一个抽象基类：`from_quantized`、`forward` 和权重字节统计。后端通过一个简单工厂选择。

`qslab/runtime/backends/gemlite.py` 是正式后端：

1. 加载时将对称整数偏移到 unsigned int4，调用移植打包函数；scale/zero 转为 kernel 所需布局。
2. packed/scales/zeros/input_scale 全部注册为 buffer，forward 每次读取当前 buffer，支持 CPU/GPU 移动；计算 dtype 在构造时指定。
3. 替换前释放原浮点投影；模型不保留完整浮点副本或离线整数张量。
4. forward 按需进行 AWQ 输入缩放，然后调用融合算子，不生成完整反量化权重矩阵。

`qslab/kernels/gemlite` 仅包含 GemLite 0.4.8 的打包、三条 FP16/BF16×W4 路径和必要工具。M=1 使用 revsplitK GEMV，2≤M≤64 使用 splitK GEMM，M>64 使用 GEMM。源码、许可证及来源见该目录 README。

HF 模型与项目引擎都调用同一个替换函数。引擎通过 `quantization` 和 `quant_backend` 配置后端；本轮的性能与 PPL 验证使用 HF，普通 FP16 KV cache。

## 验证入口

`benchmarks/w4a16` 提供单算子原包比较与小规模模型对照；`tests/quant` 验证产物、替换、移动与显存释放。所有命令和口径见 benchmark 中文 README。
