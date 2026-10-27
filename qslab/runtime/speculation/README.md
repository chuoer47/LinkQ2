# 投机推理执行入口

本目录负责依赖模型、KV 和 CUDA Graph 的投机执行。纯 token 提议与接受算法位于 `qslab/speculation`，不依赖本目录。

| 文件 | 职责 |
|---|---|
| `factory.py` | 读取草稿配置、检查目标匹配、选择 proposer |
| `draft.py` | 自回归小模型草稿，复用 ModelRunner 和分页 KV |
| `dflash.py` | DFlash 提议、目标特征缓存、请求生命周期 |
| `dflash_graph.py` | DFlash 专用 StaticCache 槽位与草稿 Graph |
| `__init__.py` | 共用的 DraftProposal 返回值 |

## DFlash

```python
from qslab.api import LLM, SamplingParams

with LLM(
    'models/Qwen3-8B-W4-rtn',
    kv_cache_dtype='kv4', smooth_kv='datasets/kv4/qwen3-8b.pt',
    spec='dflash', draft='models/Qwen3-8B-DFlash-b16',
    enforce_eager=False, enable_chunked_prefill=True,
    max_num_seqs=1, max_model_len=1024, max_num_batched_tokens=64,
    num_kvcache_blocks=8,
) as llm:
    prompt = llm.tokenizer.apply_chat_template(
        [{'role': 'user', 'content': 'Explain how binary search works.'}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )
    result = llm.generate(prompt, SamplingParams(max_tokens=256))
```

调用顺序为 `LLM → LLMEngine → factory.create_proposer → DFlashProposer`。每轮先提议，随后目标 `ModelRunner.verify_logits` 验证，由接受算法与调度器提交结果；引擎仅将已确认且已计算的目标特征交给草稿。

官方模型主体由 checkpoint 的 `auto_map` 加载：`models/Qwen3-8B-DFlash-b16/dflash.py` 中的 `DFlashDraftModel.forward`。项目复用这个 forward，调度、验证与提交使用自己的引擎。草稿权重和 KV 均为 BF16，复用目标 embedding/lm_head。

`enforce_eager=False` 开启目标及草稿 Graph；`dflash_graph=False` 只关闭 DFlash 草稿 Graph。block_size 从 checkpoint 读取，b16 为 15 个候选与 M=16 目标验证。首次长提示词提议走 eager，后续使用固定形状的草稿 Graph。DFlash 当前仅支持 greedy、非 thinking 输入。

CLI 与 HTTP 启动入口共用模型选项，例如：

```bash
python -m qslab.api.server --model models/Qwen3-8B-W4-rtn \
  --kv-cache-dtype kv4 --smooth-kv datasets/kv4/qwen3-8b.pt \
  --spec dflash --draft models/Qwen3-8B-DFlash-b16 \
  --cuda-graph --chunked-prefill --max-num-batched-tokens 64 \
  --max-num-seqs 1 --max-model-len 1024
```

HTTP 的 DFlash messages 输入自动使用非 thinking 模板。CLI `--prompt` 和 Python `generate` 的字符串输入按原始提示词处理，聊天输入应先套用上述模板。

投机推理的用法与结果见 [DFlash benchmark](../../../benchmarks/speculation/dflash/README.md)，包含固定数学、代码、聊天任务的对照。
