# 完整 checkpoint 新链路验证

日期：2026-10-15。源模型：`models/Qwen3-8B`，原始 BF16，5 个分片，399 个权重。
Python 3.10、torch 2.7.1+cu126、Triton 3.3.1，项目专属 `.conda` 环境，RTX 4090D。

## 生成

先删除局部量化包及对应产物缓存，然后通过正式 CLI 从原始 BF16 checkpoint 重新量化并写出完整包。
RTN 无需校准；AWQ 重新对 C4 英文训练集的 8×512 token 收集激活。
生成耗时、完整命令和逐层量化日志记录于实验结果目录。

## 结构

| checkpoint | 分片 | 统一索引条目 | INT8 权重 | 保留的浮点权重 |
|---|---:|---:|---:|---:|
| Qwen3-8B-W4-rtn | 5 | 651 | 252 | 147 |
| Qwen3-8B-W4-awq | 5 | 903 | 252 | 147 |

原始 399 个权重的名称与分片映射全部保留。147 个浮点权重的数值、BF16 dtype 与原始模型一致。
输出文件集合与原模型一致；仅修改 `config.json`、原统一索引及权重分片。tokenizer、生成配置等辅助文件逐字节一致。
量化说明只有 `bits`、`group_size`、`symmetric` 三项，位于原 `config.json` 内。

## 独立加载与精度

HF 离线模式，仅将新 checkpoint 目录传给加载器。PPL 使用 WikiText-2 测试集固定 8×512 token，共 4088 个预测目标；与 AWQ 校准数据分离。
普通 HF cache 的短 decode 使用 32 token 上下文、8 个后续输入 token。

| checkpoint | fake PPL | real PPL | real−fake |
|---|---:|---:|---:|
| RTN | 13.310256 | 13.310287 | +0.000031 |
| AWQ | 14.593966 | 14.601539 | +0.007574 |

两种 real 模式的 252 个量化层均执行，短 decode 的 real/fake top-1 一致率均为 100%。
模型释放后 allocated 均回到 8.125 MiB 的 cuBLAS 工作区基线。
该小样本用于验证新包及 real/fake 一致性，不代表完整语料质量结论；本轮未做引擎端到端 GPU 测试。

14 项量化/完整包测试、7 项 API 测试、4 项注释检查通过。测试包含删除临时原模型目录后独立加载 tokenizer 和模型。

原始记录：`results/checkpoint/2026-10-15/`。完整包验证命令：

```bash
python -m benchmarks.w4a16.validate_checkpoint --gpu 1 \
  --checkpoints models/Qwen3-8B-W4-rtn models/Qwen3-8B-W4-awq
```
