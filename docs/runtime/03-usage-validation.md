# Runtime 使用、服务入口与验证

以下命令均在远程 Linux/CUDA 仓库根目录、项目 `.conda` 中运行。使用已有模型目录；KV4 需对应模型的校准文件。环境建立方法见[根 README](../../README.md)。当前运行时为单 GPU，使用第一张可见卡；应在启动前设置 CUDA_VISIBLE_DEVICES，或用 CLI 的 `--device`。

## 1. Python：先跑最小生成，再组合配置

```python
from qslab.api import LLM, SamplingParams

with LLM("models/Qwen3-8B-W4-rtn",
         max_num_seqs=1, max_model_len=1024) as llm:
    result = llm.generate("The capital of France is", SamplingParams(max_tokens=16))
    print(result["text"])
    print(result["token_ids"])
```

模型目录决定浮点或 W4 权重；默认 FP16 KV、eager。启用 KV4 时增加 `kv_cache_dtype="kv4", smooth_kv="datasets/kv4/qwen3-8b.pt"`。启用 Graph 时传 `enforce_eager=False`；混合分块传 `enable_chunked_prefill=True, max_num_batched_tokens=64`。这些配置互不替代。Graph 的设计与适用路径见[专篇](04-cuda-graph.md)。

`generate_batch` 接受 prompt 列表并使用同一调度器，结果按请求顺序返回。公开 SamplingParams 默认 greedy、最多 128 token；当前主要参数为 temperature、max_tokens、ignore_eos，不应假定支持完整第三方服务的采样参数集合。

聊天输入应先调用 tokenizer 的 `apply_chat_template` 生成 token IDs；`generate` 的原始字符串不自动套模板。DFlash 必须使用非 thinking 输入，例子见[模型接入文档](../speculative/02-models-and-state.md)。

## 2. CLI

```bash
conda activate "$PWD/.conda"
python -m qslab.api.cli generate \
  --model models/Qwen3-8B-W4-rtn --device cuda:1 \
  --prompt 'The capital of France is' --max-tokens 16 \
  --kv-cache-dtype kv4 --smooth-kv datasets/kv4/qwen3-8b.pt \
  --cuda-graph --chunked-prefill --max-num-batched-tokens 64 \
  --max-num-seqs 1 --max-model-len 1024
```

`--cuda-graph` 显式开启 Graph，不传时为 eager。此例 `--device cuda:1` 将物理卡 1 映射成进程内 cuda:0。可用 `--stats-only` 输出投机统计，但未启用投机时统计为空，不代表一般吞吐指标。

## 3. HTTP 服务

```bash
python -m qslab.api.server \
  --model models/Qwen3-8B-W4-rtn --device cuda:1 \
  --host 127.0.0.1 --port 8000 \
  --max-num-seqs 4 --max-model-len 1024 \
  --cuda-graph --chunked-prefill --max-num-batched-tokens 64
```

支持 `/health`、`/v1/models`、`/v1/completions` 和 `/v1/chat/completions`；生成接口可返回普通 JSON 或 SSE。具有这些路由不代表实现了完整 OpenAI 服务协议。

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Explain binary search briefly."}],"max_tokens":32,"temperature":0}'
```

服务的后台执行线程串行推进引擎，异步 HTTP 层处理连接和事件分发。断开连接会取消请求；执行线程失败会通知等待客户端，不能让客户端无限等待。单个 LLM 实例的全局执行上下文不应被任意多个线程同时直接驱动。

## 4. 验证入口与测量说明

```bash
CUDA_VISIBLE_DEVICES=1 python -m pytest \
  tests/runtime tests/speculation -q
python -m benchmarks.runtime.engine.run --gpu 1
python -m benchmarks.runtime.cuda_graph.run --gpu 1
python -m benchmarks.runtime.chunked_prefill.run --gpu 1
```

第一条包含 GPU 测试，需要 CUDA；仅运行 CPU 测试时增加 `-m "not gpu"`。完整模型 benchmark 需要 README 指定的权重、校准和固定 token 文件。应分别核对请求结束、等待/运行取消、异常、上下文边界、页归还、Graph 实际重放及 close 后显存。

设输出数量为 $N>1$：

$$
\mathrm{TTFT}=t_1-t_{\mathrm{start}},\qquad
\mathrm{TPOT}=\frac{t_N-t_1}{N-1}.
$$

请求总耗时、首 token 延迟、首 token 后平均间隔和输出吞吐不是同一指标。对投机生成，一轮可能同时交付多个 token，要按实际交付数量统计。

[引擎](../../benchmarks/runtime/engine/README.md)、[Graph](../../benchmarks/runtime/cuda_graph/README.md)和[分块 prefill](../../benchmarks/runtime/chunked_prefill/README.md)分别测量权重/KV 组合、图重放和请求调度的效果。
