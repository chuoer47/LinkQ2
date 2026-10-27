# LinkQ2

面向 RTX 4090 系列的 Qwen3 单 GPU 推理项目，围绕 W4A16 权重量化、KV4 缓存和投机推理，实现离线压缩、GPU 算子、在线执行及可复现测量。

[设计文档](docs/README.md) · [Benchmark](benchmarks/README.md) · [正确性测试](tests/README.md)

## 项目包含什么

| 模块 | 内容 |
|---|---|
| 离线压缩 | RTN、RTN-Clip、AWQ、AWQ-Affine；输出可独立加载的完整 checkpoint |
| Kernel | GemLite W4A16 融合计算，以及 KV4 写入、分页 decode、多位置 verify 和前缀恢复 |
| Runtime | 单卡 Qwen3 执行、分页 KV、前缀复用、分块 prefill、CUDA Graph；提供 Python、CLI 和 HTTP 入口 |
| 投机推理 | N-gram、lookahead、自回归草稿和 DFlash；包含候选验证、接受及缓存状态管理 |

完整模型测量使用 Qwen3-8B，自回归草稿使用 Qwen3-1.7B，DFlash 使用匹配的官方 b16 草稿。测量条件和具体数据见各 benchmark 的说明与结果表。

## 环境准备

GPU 执行使用 Linux、NVIDIA GPU 和 CUDA 开发环境。项目环境采用 Python 3.10、PyTorch 2.7.1 / CUDA 12.6、Triton 3.3.1 和 FlashAttention 2.8.3。CUDA 工具链需提前可用，FlashAttention 的安装要求见[官方说明](https://github.com/Dao-AILab/flash-attention#installation-and-features)。

```bash
git clone https://github.com/chuoer47/LinkQ2.git
cd LinkQ2

conda env create --prefix ./.conda -f environment.yml
conda activate "$PWD/.conda"
python -m pip install -e .

python -m pip install packaging psutil ninja
MAX_JOBS=4 python -m pip install flash-attn==2.8.3 --no-build-isolation
```

Python 依赖均安装在项目专属 `.conda` 中。下列命令在仓库根目录执行；选择实际使用的 GPU，进程内对应 `cuda:0`：

```bash
export CUDA_VISIBLE_DEVICES=1
```

## 从模型到一次生成

### 1. 准备原始模型

```bash
hf download Qwen/Qwen3-8B --local-dir models/Qwen3-8B
```

模型权重放在 `models/`，校准文件放在 `datasets/`，测量输出放在 `results/`。模型与校准数据需要自行准备，相关格式与生成方法见[量化包文档](docs/quantization/03-checkpoint.md)。

### 2. 生成完整 W4 权重包

RTN 可以直接从原始模型生成，无需激活校准：

```bash
python -m qslab.compression \
  --model models/Qwen3-8B \
  --algorithm rtn_w4a16 \
  --out models/Qwen3-8B-W4-rtn
```

输出目录需要尚不存在。权重包沿用原权重名和分片，包含配置、tokenizer、未量化参数及量化辅助张量。离线保存逻辑整数，在线后端加载时打包为紧凑 int4；后续加载只需要产出目录。

RTN-Clip、AWQ 和 AWQ-Affine 的用法见[离线量化](docs/quantization/01-w4a16.md)，其中两种 AWQ 方案需要校准 token。

### 3. 使用 Python 生成

在同一环境中运行：

```python
from qslab.api import LLM, SamplingParams

with LLM(
    "models/Qwen3-8B-W4-rtn",
    max_num_seqs=1,
    max_model_len=1024,
    num_kvcache_blocks=4,
) as llm:
    result = llm.generate(
        "The capital of France is",
        SamplingParams(max_tokens=32, temperature=0),
    )
    print(result["text"])
```

默认使用 FP16 KV 和 eager 执行。原始字符串直接编码；聊天输入通过 tokenizer 的 `apply_chat_template` 准备，示例见 [Runtime 使用说明](docs/runtime/03-usage-validation.md)。

## 组合 KV4、Graph 与投机推理

准备好对应模型的 KV4 校准文件后，可以在 `LLM` 中组合以下配置：

| 配置 | 作用 |
|---|---|
| `kv_cache_dtype="kv4", smooth_kv="datasets/kv4/qwen3-8b.pt"` | 使用 KV4 缓存及该模型的平滑参数、静态 K scale |
| `enforce_eager=False` | 捕获并重放 decode / verify CUDA Graph |
| `enable_chunked_prefill=True, max_num_batched_tokens=64` | 使用 decode 优先的混合分块 prefill |
| `spec="ngram"` | 从已有 token 历史中提出候选 |
| `spec="draft", draft="models/Qwen3-1.7B-W4-rtn"` | 使用普通自回归草稿模型 |
| `spec="dflash", draft="models/Qwen3-8B-DFlash-b16"` | 使用官方 DFlash 草稿 |

目标与普通草稿分别使用自身的 KV 校准和缓存。DFlash 草稿保持 BF16 权重与 KV，使用 greedy、非 thinking 输入。lookahead 使用方式及各组合条件见[投机推理文档](docs/speculative/README.md)。

KV4 校准方法见 [KV4 方案](docs/quantization/02-kv4.md)，图族、填充和图外执行见 [CUDA Graph 设计](docs/runtime/04-cuda-graph.md)。完整 Python、CLI 和 HTTP 示例见 [Runtime](docs/runtime/README.md)。

## 测试与 Benchmark

测试关注离线算法及权重包、Kernel 数值、运行状态和投机接受机制。性能测量则单独记录 PPL、延迟、吞吐、接受率和显存。

```bash
# CPU 测试
python -m pytest tests -m "not gpu" -q

# CPU 与 GPU 测试
python -m pytest tests -q

# 安装进程显存采样依赖，并运行一组完整引擎测量
python -m pip install -e ".[benchmark]"
python -m benchmarks.runtime.engine.run --gpu 1
```

各 benchmark 有独立目录、一键入口、中文说明和默认小规模输入。运行时生成 JSON 指标与表格，并同步更新对应结果文档。入口与准备条件见 [Benchmark 目录](benchmarks/README.md)，覆盖范围见 [测试说明](tests/README.md)。

## 继续阅读

| 入口 | 阅读内容 |
|---|---|
| [量化与离线包](docs/quantization/README.md) | 算法数学、校准、权重包字段与加载约定 |
| [Kernel 设计](docs/kernels/README.md) | W4A16 计算路径、KV4 Attention 的布局、复用和并行取舍 |
| [Runtime](docs/runtime/README.md) | 加载、请求调度、KV 页、CUDA Graph、接口与生命周期 |
| [投机推理](docs/speculative/README.md) | 候选生成、接受概率、草稿同步与性能取舍 |
| [项目架构](docs/PROJECT_MAINLINE.md) | 模块职责及代码导航 |

`qslab/` 保存核心实现，`adapters/` 负责 Hugging Face 模型构造与参考加载，`scripts/` 提供数据和校准工具。原理说明、正确性测试和真实模型测量分别维护在 `docs/`、`tests/`、`benchmarks/`。

## 许可证与来源

项目原创部分采用 [MIT License](LICENSE)，版权署名为 chuoer47。

| 第三方代码 | 许可证与来源 |
|---|---|
| GemLite W4A16 算子 | [Apache 2.0](qslab/kernels/gemlite/LICENSE)，见[源码说明](qslab/kernels/gemlite/README.md) |
| nano-vLLM Runtime 与 FP16 KV 写入 | [MIT](qslab/runtime/LICENSE.nano-vllm)，见[接入说明](qslab/runtime/UPSTREAM.md) |

第三方代码保留各自的许可证和版权声明；模型权重、数据及外部依赖使用各自来源的许可证。
