# LinkQ2 项目主线与 W4A16 链路梳理

核对日期：2026-10-15。本文件记录当前源码和已有实验反映的状态；本次没有重新执行 GPU 测试，没有搬迁或修改算子。

## 项目目标

在 RTX 4090/4090D 上，验证 W4A16 权重量化、KV 量化的质量、显存和速度收益；以这些能力支撑一个简易的 vLLM 类推理引擎，再验证投机推理。每项能力必须有独立测试、中文说明和一键 benchmark，默认使用小规模数据。

```text
原始浮点模型
    │
    ├─ 离线权重量化 → 逻辑量化产物 → 算子专用打包 → W4 Linear
    │                                              │
    └─ 浮点 Linear 基线 ────────────────────────────┤
                                                   ↓
                                      模型执行：HF 验证 / 自己的引擎
                                                   │
                                      KV16 基线 / 量化 KV + Attention
                                                   │
                                      调度、批处理、缓存管理、图执行
                                                   │
                                      普通生成 / 投机生成与验证
```

权重量化算法、执行算子和推理引擎分别验收。更换算子时复用同一份逻辑量化产物，避免重新量化混入对照实验。

## 当前 W4A16 主链路

| 环节 | 当前代码 | 输入与输出 |
|---|---|---|
| 离线量化 | `qslab/compression/pipeline.py`、`algorithms/w4.py` | 浮点权重、可选校准激活 → q、scale、zero、可选 input_scale |
| 逻辑产物 | `qslab/artifacts/schema.py`、`io.py` | 与具体 kernel 无关的整数及元数据；目前 values 使用 int8 容器，还不是紧凑 int4 |
| 通用 W4 打包 | `qslab/conversion/export_qslab_w4.py`、`qslab_w4.py` | 逻辑整数 → 每个 uint32 存放 8 个 int4，以及 scale 等 |
| CUDA 执行 | `qslab/models/w4linear.py`、`runtime/backends/w4.py` | 自定义 CUDA 后端或 Marlin 后端；Marlin 另有布局重排 |
| GemLite 执行 | `benchmarks/fused_quant/adapter.py` | 直接从逻辑产物重打包，调用已安装的 GemLite；目前只用于 benchmark |
| 模型验证 | `benchmarks/real_quant/`、`fused_quant/` | 替换 HF Linear，验证覆盖、PPL、prefill、decode 和显存 |
| 自己的引擎 | `qslab/runtime/model/loader.py`、`execute/model_runner.py` | 已有 W4 权重替换入口；GemLite 尚未注册为运行时后端 |

fake 路径离线展开成浮点权重，forward 使用原生浮点 Linear。real 路径保留紧凑 W4 权重，在融合算子内部按块反量化并运算，不常驻整份展开权重。

## 第三方算子的实际依赖边界

### Marlin：源码已在仓库内

`qslab/kernels/marlin/` 已包含：

- `csrc/kernel.cu`：上游 CUDA kernel。
- `csrc/binding.cpp`：PyTorch/C++ 绑定。
- `build.py`：编译并加载扩展。
- `ops.py`：布局重排、scale 排列、调用扩展。
- `__init__.py`、`LICENSE`：入口与来源许可。

这条链路并不需要安装整个 Marlin Python 包。CUDA 算子从 Python 调用时，仍需要绑定、构建脚本和打包规则；只复制一个 `.cu` 文件不足以完成接入。源码原始版权声明与 LICENSE 应随提取代码保留。

现有自定义 CUDA 实现位于 `qslab/kernels/w4a16/`，也是 CUDA 源码、绑定、加载入口的组合，应与 Marlin 分开标注来源和测试结果。

### GemLite：目前仍导入整包

当前依赖位于项目 `.venvs/gemlite-0.4.8/`，`FusedLinear` 仍执行 `import gemlite`，包括 direct 模式。direct 只绕过了模块的调度入口，没有去掉整库依赖。

已核对 0.4.8 当前 W4 路由：

| M，即一次 Linear 的输入行数 | 实际路径 | 候选提取文件，均在上游 `gemlite/triton_kernels/` |
|---|---|---|
| 1 | GEMV_REVSPLITK | `gemv_revsplitK_A16fWnO16f_int32packing.py` |
| 2–64 | GEMM_SPLITK | `gemm_splitK_A16fWnO16f_int32packing.py` |
| >64 | GEMM | `gemm_A16fWnO16f_int32packing.py` |

还需要 `utils.py`、`config.py`、类型定义以及离线打包和主机调用代码。上游 kernel 的调优逻辑还引用 `core.GEMLITE_TRITON_CONFIG_CACHE`，配置重载入口还引用其他 kernel。因此这是一份候选提取清单，尚未经过裁剪后的闭包验证，不能宣称复制这三个文件就能独立运行。

提取时保留上游 kernel 数学计算和来源，在本项目提供小的打包、路由、配置适配层。HQQ 辅助封装、其他精度的高层转换入口、其他 GPU 的调优预设不应成为主链路的必需依赖。最终保留范围以独立导入、首次编译/调优、正确性、完整调用测试为准。

移除 GemLite 整包依赖后，Triton 实现仍然需要兼容的 PyTorch、Triton 和 CUDA 驱动。

## 当前证据与待补项

| 能力 | 已有代码/证据 | 尚需验收或收口 |
|---|---|---|
| W4A16 | 已有量化、打包、CUDA 后端；GemLite HF 小样本实验验证了显存收益与 real/fake PPL 接近，部分形状有 GPU 时间收益 | GemLite 最小源码接入；统一后端入口；普通 HF 的端到端速度尚未超过浮点基线 |
| KV 量化 | `kernels/kv4/` 已有写入、布局、paged attention；已有相应测试文件 | 独立的一键 benchmark、KV16 配对基线、不同长度的误差/显存/性能结果 |
| 简易引擎 | 已有模型执行、调度、block manager、图执行和 API | 用统一基线分别验收浮点、W4、量化 KV 和二者组合，避免同时改变多个因素 |
| 投机推理 | 已有 ngram、lookahead、draft 相关代码与测试 | 独立 benchmark；验收采样正确性、接受率、实际提交 token/s、额外显存与端到端加速比 |

存在测试文件只说明已有测试实现，本次没有重新运行它们，不能据此认定相关能力已通过验收。

注意：当前 `w4.auto` 的 `CROSSOVER_M=0`，Marlin 可用时会优先走它，不是根据 shape 的实测性能动态选最快实现。此前 Marlin 在较大 M 的对照中存在数值失败，正式默认后端必须先重新满足正确性验收。

## 环境收口

- 现有 `qslab` conda 是 PyTorch 2.5.1+cu124 / Triton 3.1.0；CUDA 构建还使用项目对应的 nvcc/GCC 工具链。
- GemLite 实验使用继承原 conda 包的项目 venv，在其中覆盖为 Triton 3.3.0。它是实验环境；PyTorch 2.5.1 的包元数据固定 Triton 3.1，不能直接把这个覆盖组合当作长期统一环境。
- CUDA 算子环境保留 PyTorch、编译工具链和构建依赖；Triton 算子环境保留配套 PyTorch/Triton。HF、数据集和评测依赖按需加入；引擎的 FlashAttention 依赖单独记录。
- 目标是可重建的项目环境和明确的版本文件。先完成依赖与后端选择，再替换临时环境；本次不删除已有环境、模型、缓存和实验记录。

## 每项功能的一键验收约定

目标入口为 `benchmarks/w4a16/`、`kv_quant/`、`engine/`、`spec_decode/`。每个目录至少包含中文 `README.md` 和 `run.py`，提供默认小规模的运行配置。公共计时、数据选择、环境记录放入共享 benchmark 工具，生产功能放入 `qslab/`；不由一个 benchmark 导入另一份实验专用实现。

每次运行输出到 `results/<功能>/<run_id>/`，保存配置、源码与依赖版本、完整命令、输入哈希、原始数据、日志、JSON/CSV 汇总及中文结论。首次构建/调优与稳态计时分开；GPU 图回放时间与普通调用/整模时间分开。

W4A16 至少分别验收：

1. 逻辑权重及打包正确性；算子与高精度参考比较。
2. 实际模型矩阵形状的浮点/fake/real 延迟，覆盖 M=1 与短 prefill。
3. 常驻、峰值显存及真实 packed 权重存储，确认未保留整份浮点副本。
4. 固定小样本配对 PPL、HF prefill/decode；接入引擎后另报引擎测量。

历史 benchmark 和结果先作为证据保留。统一入口稳定后再归档旧实验，避免靠删除文件来掩盖未解决的问题。

## 下一步顺序

1. 先完成 W4A16 最小源码提取和调用闭环：选一个已有数值证据的候选，保持同一份量化权重和参考，验证提取前后输出、存储与延迟。
2. 将验证后的算子接入统一 W4 后端，建立一个默认小规模的一键 benchmark。
3. 同样验收 KV 量化，再验收引擎中的 W4/KV 独立开关及组合。
4. 在稳定的非投机基线上验收投机推理，不把已有原型重新从零实现。
