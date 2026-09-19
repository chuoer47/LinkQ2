# 11 — 服务面 v1（`qslab.api.server`）

这一目录不是吞吐实验，是**服务面的验收脚本 + 读数底稿**。
被测代码：`qslab/api/server.py`（OpenAI 兼容最小面，跑在 aiohttp 上；aiohttp 以
optional extra `.[serve]` 声明，核心包不引入 web 依赖）。
底稿：`results/m12_service_smoke.txt`（2026-09-19，GPU1，Qwen3-1.7B fp16 权重 + KV4）。
测试：`tests/test_api_server.py`（7 条，快档，不占 GPU）。

## 端点

| 端点 | 行为 |
|---|---|
| `GET /health` | `{status, model, active_requests}` |
| `GET /v1/models` | 列当前加载的模型 |
| `POST /v1/chat/completions` | `messages`（走 tokenizer 的 chat template）或 `prompt`；`stream` 支持 SSE |
| `POST /v1/completions` | 原始 prompt，同上 |

## 为什么长这样

- **一个泵线程独占引擎**。`LLMEngine.step()` 是同步单线程的，批是在它内部的 scheduler 里
  成的，所以服务**不按请求开线程**（那会让 N 个线程抢一个非线程安全的引擎）。客户端把 Job
  投进 `EnginePump` 的收件箱，泵用 `call_soon_threadsafe` 往各请求自己的 asyncio 队列喂增量。
- **逐 token 流式需要 step 级可见性**，而 `step()` 只报告已完成的序列。所以 `add_request`
  现在**返回 Sequence**（本轮唯一的引擎侧改动），泵每步 diff `seq.completion_token_ids`。
  被抢占后重算 prefill 的序列，其 token 列表单调增长，diff 依旧成立。
- **断连不取消**（已知限制，见 TODO 14）：客户端跑了，请求会继续生成到 token 预算用完。

## 复现

```bash
bash /tmp/qsl_run.sh "python -u -m qslab.api.server --model models/Qwen3-1.7B \
  --device 1 --host 127.0.0.1 --port 8077 --max-num-seqs 16" &   # --util 不给即 0.9
# 等 /health 通了再打（本轮 3 秒）
bash benchmarks/11-service-surface/smoke.sh
bash benchmarks/11-service-surface/smoke_concurrency.sh
python -m pytest tests/test_api_server.py -q                     # 7 passed
```

⚠ 收尾要按 PID 精确 kill。用 `pkill -f qslab.api.server` 会**匹配到发起命令自己的命令行**，
本轮第一次采集因此把 ssh 会话一起打死，底稿里只剩空头（这就是 `results/m12_service_smoke.txt`
重做过一次的原因）。

## 读数

- `/health` 3 秒内通；非流式 24 token 正常回（`finish_reason=length`，
  `usage{13,24,37}`）；SSE 分帧 + `[DONE]` 正常；`{"max_tokens":0}` → **400**。
- **8 条并发、每条 48 token：384 tokens / 0.4439 s 墙钟 = 865 tok/s**，8 条同时完成
  ⇒ 它们确实进了同一个批。这一步用来印证"服务面没有把批量拆成串行"。
- 对照 `results/m12_batch_1.7b.txt` 的 bs=8 行（1014.9 tok/s 墙钟）：低 14.8%。
  **这 14.8% 不能整笔记在 HTTP 上**——两边 prompt 长度不同（22 vs 33 token，prefill 占比不同）、
  curl 进程启动算进了墙钟、服务端 `--util 0.9` 而扫描是 0.5、这一轮还没带 `--smooth-kv`
  （`bench_batch.py` 会自动找 calib 文件）。所以这一格的结论只到
  **"服务面与进程内扫描同数量级、批确实成了"**，不到"HTTP 层开销 15%"。
- Qwen3 的 chat template 默认带 thinking 前缀 ⇒ `messages` 路径第一个 delta 就是 `<think>`。
  **服务面不解析、不过滤**，那是产品层的事；写在这里免得被读成引擎 bug。
- `max_tokens: 0` 那条是**测试先抓出来的真 bug**：原来写 `body.get("max_tokens") or DEFAULT`
  会把 0 静默吃成默认值 128。修法是先显式判 `None`（`or` 挡不住 0）。

## 已登记的限制（都在 TODO，不是"稍后顺手做"）

| 限制 | 后果 | TODO |
|---|---|---|
| 断连不取消 | 底稿里就有一次现形：`smoke.sh` 的 `head -6` 提前断管，服务端抛 `ClientConnectionResetError: Cannot write to closing transport`（`server.py:256`）。目前这个异常只脏日志，不影响其他请求 | 14 |
| 无鉴权、无并发闸 | `active_requests` 只是观测值，不设上限 ⇒ 打满只会排队 + 抢占，不拒绝 | 15 |
| 无 W4/服务化的吞吐标定 | 本轮只测 1.7B fp16 权重；8B W4 起服路径同一套代码，但没实跑过 HTTP 面 | 16 |
