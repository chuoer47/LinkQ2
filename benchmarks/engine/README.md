# 单卡推理引擎第一版

范围：单卡 Qwen3、普通 eager 推理；FP16/W4 权重与 FP16/KV4 缓存独立选择。默认不使用 CUDA graph、compile 或投机推理。

## 一键运行

在仓库根目录，使用项目环境：

```bash
.conda/bin/python -m benchmarks.engine.run --gpu 1
```

需要已有 `models/Qwen3-8B`、完整 `models/Qwen3-8B-W4-rtn` checkpoint、`datasets/kv4/qwen3-8b.pt` 校准文件，以及 `datasets/w4a16/samples.json`。默认复用这些产物，不重新量化或校准，不安装依赖。

只跑 FP16 基线：

```bash
.conda/bin/python -m benchmarks.engine.run --gpu 1 --modes fp16-fp16
```

可用 `--w4-model models/Qwen3-8B-W4-awq` 改测 AWQ 完整包。`--out` 指定结果目录。

## 测什么

- 四种组合：`fp16-fp16`、`fp16-kv4`、`w4-fp16`、`w4-kv4`。每种配置独立进程、顺序运行，固定 4 页×256 token KV 容量。
- 执行正确性：3 个短提示，各生成 16 token；原始权重对照 HF FP16，W4 对照相同产物的 HF fake。先比较自由生成，再固定参考历史比较全部 48 个位置的 logits、top1 和参考 token NLL 差异。
- FP16 KV 的 logits 相对 L2 须不超过 0.02、teacher-forced top1 一致率至少 95%；KV4 的对应门槛为 0.3 和 80%，用于发现明显的集成错误。这些是小样本验收阈值，不是精度保证；不替代 W4/KV4 已有 PPL benchmark，也不把自由生成分叉直接视为算子错误。
- 生命周期：空输入、超长输入拒绝；上下文边界截断并结束；等待与运行中的取消；正常完成后页面引用归零；关闭引擎后释放模型和池。执行异常与 HTTP pump 回收另有回归测试。
- 性能：WikiText 固定首段取 128/512 token，各生成 16 token；每种形状完整预热 1 次，测 3 次取中位数。关闭前缀复用，保证每次完整 prefill。
- 计时：同步主机墙钟，从 token ID 请求入队至最后 token 提交，包含调度、输入张量准备、模型、KV、lm_head、采样、回收；不含加载、tokenize/detokenize、HTTP 和参考对照。
- 指标：TTFT、TPOT、整次请求延迟、输出 token/s、相对 FP16 基线的 TPOT 加速比，以及权重/固定 KV 字节数、常驻 allocated、启动及运行峰值 allocated、close 后 allocated。

关闭后的显存与预先初始化的 cuBLAS 工作区基线比较，允许 1 MiB 测量余量。当前环境 cuBLAS 保留 8.125 MiB；已通过独立检查确认模型关闭后无存活 CUDA Tensor，清理 cuBLAS 工作区后 allocated 归零。正式推理不主动清理进程共享的库工作区。

默认规模很小；本轮不验证多请求公平性、抢占压力、CUDA graph 或投机加速。FP16 KV 目前只支持 eager，分页读写使用官方 FlashAttention 接口，未修改第三方算子的计算实现。

## 结果与复现

`results/engine/<时间>/` 保存 `run.json`、固定输入、两份 HF 参考、四份原始 JSON、运行日志、`summary.json` 和中文 `report.md`。`run.json` 记录代码提交、工作区状态、参数、环境路径与随机种子。

普通使用：

```python
from qslab.api import LLM, SamplingParams

with LLM("models/Qwen3-8B", max_num_seqs=4) as llm:
    result = llm.generate("The capital of France is", SamplingParams(max_tokens=16))

with LLM("models/Qwen3-8B-W4-rtn", kv_cache_dtype="kv4",
         smooth_kv="datasets/kv4/qwen3-8b.pt", max_num_seqs=4) as llm:
    result = llm.generate("The capital of France is", SamplingParams(max_tokens=16))
```

权重模式由 checkpoint 配置决定，KV 模式由 `kv_cache_dtype` 决定。KV4 要求校准文件；FP16 KV 不需要。`close()` 可重复调用；请求完成或取消时归还页面，已预分配的池在关闭引擎时释放。
