# 算法策略与显式量化层清单验证

日期：2026-10-15。源模型：`models/Qwen3-8B`，原始 BF16，5 个分片，399 个权重。
Python 3.10、torch 2.7.1+cu126、Triton 3.3.1，项目专属 `.conda` 环境，RTX 4090D。

## 生成

先删除上一版 RTN、AWQ 完整包，再通过正式 CLI 从原始 BF16 checkpoint 重新量化并写出新包。
RTN 无需校准；AWQ 重新对 C4 英文训练集的 8×512 token 收集激活。
RTN 生成耗时 15.95 秒，AWQ 55.22 秒。完整命令和逐层量化日志记录于实验结果目录。

## 结构

| checkpoint | 分片 | 统一索引条目 | INT8 权重 | 保留的浮点权重 |
|---|---:|---:|---:|---:|
| Qwen3-8B-W4-rtn | 5 | 651 | 252 | 147 |
| Qwen3-8B-W4-awq | 5 | 903 | 252 | 147 |

原始 399 个权重的名称与分片映射全部保留。147 个浮点权重的数值、BF16 dtype 与原始模型一致。
仅新增 `quantized_layers.json`，每个包明确列出 252 个模块名。tokenizer、生成配置等复制文件逐字节一致。
`config.json` 的 `quantization_config` 包含 `quant_method`、`bits`、`group_size`、`symmetric`：

```json
{"quant_method": "awq_w4a16", "bits": 4, "group_size": 128, "symmetric": true}
```

量化结果定义与写包逻辑位于 `qslab/compression/schema.py` 和 `checkpoint.py`。
算法名称集中在 `qslab/quantization.py` 的枚举中；在线不导入离线组件。
加载器先按算法名选择 `RuntimeQuantization` 的具体实现，再按 JSON 清单选择模块。
AWQ 策略明确要求 input_scale；非对称策略明确要求 zero_point，不通过字段是否存在猜测处理方式。

## 独立加载与精度

HF 离线模式，仅将新 checkpoint 目录传给加载器。本轮是小规模链路验证：WikiText-2 测试集前两段固定片段，各取 128 token，共 254 个预测目标；与 AWQ 校准数据分离。
普通 HF cache 的短 decode 使用 32 token 上下文、8 个后续输入 token。

| checkpoint | fake PPL | real PPL | real−fake |
|---|---:|---:|---:|
| RTN | 24.150264 | 24.150330 | +0.000067 |
| AWQ | 26.362033 | 26.354651 | −0.007382 |

两种 real 模式的 252 个量化层均执行，短 decode 的 real/fake top-1 一致率均为 100%。
模型释放后 allocated 均回到 8.125 MiB 的 cuBLAS 工作区基线。
该小样本用于验证新包及 real/fake 一致性，不代表完整语料质量结论。本轮未重测性能，也未做引擎端到端 GPU 测试。

16 项量化测试通过，包含四种算法、完整分片、删除临时原模型目录后的独立加载、CPU/GPU 移动及释放。
新增测试确认清空层清单后不会因存在 scale 而选择量化层，AWQ 缺少 input_scale 会报错，未知算法名会拒绝加载。
离线/在线导入隔离检查通过。

原始记录：`results/checkpoint/2026-10-15-strategies/`，包括校验汇总、包结构、构建命令/日志、测试日志及固定 token。
产出目录仍为 `models/Qwen3-8B-W4-rtn` 和 `models/Qwen3-8B-W4-awq`。

重新生成命令见根 README。默认完整包验证使用 8×512 token：

```bash
python -m benchmarks.w4a16.validate_checkpoint --gpu 1 \
  --checkpoints models/Qwen3-8B-W4-rtn models/Qwen3-8B-W4-awq
```
