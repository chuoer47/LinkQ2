# 架构

> 面向后续开发（M7 量化 paged attention、nano-vllm 整合）的架构说明。
> 分层规则的设计依据见 [design-r1.md](design-r1.md)。

## 分层与依赖

```
L4  qslab/api/       llm.py (LLM 门面)  cli.py
L3  qslab/engine/    core.py (QslabEngine)  spec/{verify,modes,graph_decoder}.py
L2  qslab/models/    loader.py  patched.py (Qwen3Attention 子类)  w4linear.py
L1  qslab/quant/     packfmt.py  w4.py  calibrate.py  quantize.py  kv4_plan.py
                     backends.py + w4_backends.py      (QuantBackend)
                     kv_strategies.py + _impl.py        (KVCacheStrategy)
                     cache/kv_cache.py                  (fp16/kv8/kv4 存储)
L0  qslab/kernels/   ops.py (v1 绑定)  marlin_ext.py  marlin_backend.py  csrc/*.cu
    adapters/        tokenizer.py  model_builder.py    ← 唯一可 import transformers
    registry.py      通用注册表（跨层共享，无依赖）
    config.py        配置 dataclass（跨层共享）
```

**规则**：每层只 import 直接下层。`adapters/` 例外，可被 L2/L3/L4 使用。

## 一次 decode step 的数据流

```
LLM.generate(prompt)                       L4
  └─ QslabEngine.generate(ids)             L3
       ├─ prefill: model(input_ids)        L2  transformers Qwen3 前向
       │            └─ PatchedQwen3Attention.forward
       │                 ├─ q/k/v 投影: W4Linear → QuantBackend.linear(x)   ← L2→L1
       │                 │     └─ w4.auto: M>8 ? marlin_gemm : w4a16_gemm   ← L1→L0
       │                 ├─ RoPE + KV 写入: engine_cache.update(k, v)       ← L1 存储
       │                 └─ attention: SDPA(k_full, v_full)                 ← 已反量化的 fp16
       └─ decode 循环: 逐步 decode_step, 每步 argmax
```

关键：**attention 只看得到 fp16 的 K/V**。量化/反量化全部封装在 `KVCacheStrategy`
返回的 cache 对象里（`update()` 的契约就是"存量化、返 fp16"）。这也是为什么
M7 的量化 paged attention 需要在 L0 新增 kernel、在 L1 新增一个 strategy 实现——
两侧都不动 L2/L3。

## 三个策略接口

| 接口 | 定义 | 实现 | 工厂 |
|---|---|---|---|
| `QuantBackend` | `quant/backends.py` | `quant/w4_backends.py` | `get_backend(name, ...)` |
| `KVCacheStrategy` | `quant/kv_strategies.py` | `quant/kv_strategies_impl.py` | `build_strategy(kv_mode, plan)` |
| `SpeculationMode` | `engine/spec/modes.py` | 同文件 | `build_mode(name)` |

注册用 `qslab/registry.py` 的 `@REGISTRY.register("name")`。

**接入注意**：注册发生在模块 import 时。`quant/__init__.py` 显式 import 了
`w4_backends`，保证 `import qslab.quant` 后注册表非空。

## 打包格式 qslab_w4_v1

```
qfp   [O, I/8]    uint32   8 个有符号 int4/nibble（LSB first），值域 [-8, 7]
scale [O, I/128]  fp16     每 128 宽一组的对称 scale
zero  [O, I/128]  fp16     v1 恒 0（非对称的预留位）
```

Marlin 后端需要另一种排布（`[I/16, O*2]` int32 + permuted scale），转换在
`quant/w4_backends.py::W4MarlinBackend.__init__` 里一次完成，置换表逐行照抄上游。

## 扩展点

| 想加什么 | 动哪里 |
|---|---|
| 新的权重量化方案（如 W8A8） | L1 加一个 `QuantBackend` 实现 + 注册 |
| 新的 KV 精度 / paged 存储 | L1 加一个 `KVCacheStrategy` 实现 + 注册（`cache/` 下加存储类） |
| 新的提案来源（如 Medusa head） | L3 加一个 `SpeculationMode` 实现 + 注册 |
| 新的模型架构 | L2 加模型定义；`adapters/` 处理 HF 侧 |
| 新的 kernel | L0 加 `.cu` + 绑定；L1 的 backend 调用它 |

任何一格都不需要改其它层的代码。
