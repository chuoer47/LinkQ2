# 10 — 与官方 vLLM 的同口径吞吐对照

脚本：`bench_vllm.py`（跑在独立的 `vllm` conda env，**不是** qslab env）。
底稿：`results/m12_vllm_1.7b.txt`（2026-09-19，4090_public GPU3）。
本 runtime 一侧的对照表来自 `../09-batch-throughput/`。

## 匹配了什么

同一张卡、同一个 fp16 `models/Qwen3-1.7B` 目录、**同一批 prompt token id**
（同一个 `seed=11` 的随机首 token + 同一段 copy prompt，与 `bench_batch.distinct_prompts`
逐字一致）、贪心、`ignore_eos`、256 输出 tokens、`max_num_seqs=32`、`gpu_memory_utilization=0.5`、
`max_model_len=4096`、CUDA graph 开、前缀缓存两边都开（都被随机首 token 打掉）。

| | vLLM 0.11.0（底稿实测） | 本 runtime（09 目录） |
|---|---|---|
| 权重 | fp16（`quantization=None`） | fp16 |
| KV | fp16，`block_size=16` | **int4**（KV4），`kvcache_block_size=128` |
| 批上限 | `max_num_seqs=32`，`max_num_batched_tokens=8192` | 32 / 16384 |
| CUDA graph | `cudagraph_mode=[2,1]`，capture sizes 1…64 | `graph_bs=[1,2,4,8,16,32]` |
| attention | Flash Attention backend | 自研 paged int4 decode |
| KV 池 | 8.10 GiB → **75,808 tokens**（18.51× @4096） | 7.27 GiB 预算 → **135,936 tokens** |

## 读数：只有 `tok/s(wall)` 一列可比

vLLM 的离线 `LLM.generate()` 只能给"整段墙钟 / 生成 token 数"，拿不到 decode 窗口。
所以对照走墙钟口径；本 runtime 的 decode-window 列只用于自己内部的对比。

| bs | qslab tok/s(wall) | vLLM tok/s(wall) | qslab / vLLM |
|---|---|---|---|
| 1 | 143.6 | 202.4 | 0.71× |
| 2 | 261.6 | 387.6 | 0.67× |
| 4 | 518.9 | 759.8 | 0.68× |
| 8 | 1014.9 | 1462.7 | 0.69× |
| 16 | 1874.2 | 2740.7 | 0.68× |
| 32 | 3516.0 | 4840.4 | 0.73× |

批量加速曲线：**qslab 24.43× vs vLLM 23.91×**（各相对自己的 bs=1，32 路）。

## 结论

1. 绝对吞吐是 vLLM 的 **0.67–0.73×**，且这个比例**不随并发变化**。两条加速曲线重合
   （24.4× vs 23.9×）⇒ 损失不是"随批量累积的调度缺陷"，形态上更像一笔**每步固定开销**。
   具体花在哪（自研 paged int4 decode kernel、Python 侧 `step()` 循环、还是 sampler）
   **未做逐层 profile，未证**。
2. 量化侧的收益是实打实的：**同 util 下 KV 池容量 135,936 vs 75,808 tokens（1.79×）**。
   按真实字节算更清楚：vLLM 的 fp16 KV 是 112 KiB/token，我们的是 28.9 KiB/token（**3.87×**），
   而现在只兑现到 1.79× 是因为 TODO 11 的 **1.94× over-billing**（计费 `2*head_dim`、实配 132 B/head/layer）。
   ⇒ 修掉 over-billing 后同一份预算约可放 **26 万 tokens**。
3. 一句话口径：**这台机器上，自研引擎在纯 fp16 场景不如 vLLM，赢的是 4-bit 带来的显存/容量**。
   任何"我们比 vLLM 快"的表述在本仓库目前没有证据。

## 这一格**不是**什么（写死，别被引用时误读）

- **不是量化对比。** vLLM 0.11 有 AWQ/GPTQ 的 W4A16 kernel，但**没有 int4 KV**；
  而本项目的权重 pack 是自研 `qslab_w4_v1`（`models/*.origin.txt`：由本机 fp16 权重打包），
  vLLM **加载不了**。所以要精度对齐只能 fp16↔fp16，KV4 在 vLLM 侧没有对应物。
- **不是 kernel 对比。** 我们的 Marlin 路径是从上游 vendor 来的、paged decode 也与 vLLM 同源，
  kernel 级胜负不是这里要回答的问题。
- **不含投机。** vLLM 侧没跑 `prompt_lookup`，所以 09 目录里"高并发下 n-gram 是负收益"
  这条只能与**自己**的 off 行比，不能拿来跟 vLLM 比。
- **没有任何逐 token 一致性检查**：4-bit 贪心是混沌的，仓库只认 PPL / 吞吐 / 接受率。

## 复现

```bash
# 独立 env（torch 2.8.0+cu128 / py3.12），不需要 qslab 的工具链变量
CUDA_VISIBLE_DEVICES=3 MODEL=./models/Qwen3-1.7B \
  TOKENS=256 BATCHES=1,2,4,8,16,32 UTIL=0.5 \
  <conda-base>/envs/vllm/bin/python -u benchmarks/10-vllm-compare/bench_vllm.py
```

三个坑（都在这次的实跑里踩过，脚本里已固化前两条）：

1. **flashinfer 采样器在 engine 初始化就炸**：`profile_run → _dummy_sampler_run →
   topk_topp_sampler.forward_cuda → flashinfer_sample`，根因是 `RuntimeError: Could not find
   nvcc and default cuda_home='/usr/local/cuda' doesn't exist`——这个 env 没有 nvcc。
   脚本顶部已 `os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")`。
   （不去装 CUDA toolkit：采样器不是被测量。）
2. **父进程失败会留下孤儿 `VLLM::EngineCore` 占着 8 GiB**。这次两次失败留下 2 个，
   下一次启动直接 `ValueError: Free memory on device (7.11/23.53 GiB)`。
   ⇒ 重跑前先 `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`，
   按 PID 精确清掉**自己**的进程（0/2 号卡是别人的常驻作业，不要碰）。
3. vLLM 0.11 的 `LLM` 对象**没有** `llm_config` 属性，配置要经 `llm.llm_engine.vllm_config`。

## 还缺的格子（都已评估过，不是忘了）

- **8B W4A16↔W4A16**：需要 vLLM 能加载的同 checkpoint 4-bit pack（`bitsandbytes` 或
  `compressed-tensors` 路线）。这要在共享的 `vllm` env 里装新依赖，属于改环境，先问再做。
  注：8B 的 fp16 权重在 `util=0.5` 下根本装不下（15.2 GiB > 11.76 GiB 预算），
  所以"vLLM 跑 fp16 8B 对我们跑 W4 8B"只能是**不等价口径**，不能当引擎对比。
- **权重常驻字节**一侧不需要这里重做：`results/m10_w4_residency.txt` 已经量过
  8B W4 常驻 3.339 GB = fp16 的 0.26×。
