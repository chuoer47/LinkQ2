# design-r1: 分层重构设计

> R1-R6 的总体设计。定稿后按阶段推进。

## 目标结构（L0-L4 五层）

```
qslab/                          ← 单包，五层子包
├── L0 原语层
│   └── qslab/kernels/           # CUDA/Triton kernel + torch 绑定
│       ├── csrc/                #   .cu 源码（从 kernels/csrc 迁入）
│       │   ├── w4a16_gemm.cu
│       │   ├── w4a16_gemm_api.cu
│       │   └── test_w4a16_gemm.cu
│       ├── ops.py               #   v1 kernel 绑定
│       ├── marlin_ext.py        #   marlin 编译
│       └── marlin_backend.py    #   marlin repack + gemm  ← 纯张量操作，属 L0
│
├── L1 量化层
│   └── qslab/quant/
│       ├── packfmt.py           #   qslab_w4_v1 格式读写
│       ├── w4.py                #   RTN/AWQ 算法
│       ├── calibrate.py         #   激活统计
│       ├── quantize.py          #   离线驱动（HF → pack）
│       ├── kv4_plan.py          #   KV 离群分析
│       ├── backends.py          #   ★ QuantBackend 抽象 + 注册表（R2）
│       └── formats/             #   （R2 可选拆分）格式定义
│
├── L2 模型层
│   └── qslab/models/
│       ├── loader.py            #   权重加载（config/safetensors）
│       ├── patched.py           #   Qwen3Attention 替换子类
│       ├── w4linear.py          #   W4 Linear（经 L1 的 backend，不直连 L0）
│       └── qwen3.py             #   （R3）模型装配入口
│
├── L3 引擎层
│   └── qslab/engine/
│       ├── core.py              #   QslabEngine（decode 循环）
│       ├── cache/               #   KV cache 策略（原 qslab/cache/）
│       │   ├── base.py
│       │   ├── fp16.py / kv8.py / kv4.py
│       │   └── strategy.py      #   ★ KVCacheStrategy 抽象 + 注册表（R2）
│       └── spec/                #   投机推理
│           ├── engine.py        #   SpeculativeEngine（原 verify.py）
│           ├── modes.py         #   ★ SpeculationMode 抽象 + 注册表（R2）
│           └── graph.py         #   原 graph_decoder.py
│
├── L4 接口层
│   └── qslab/api/
│       ├── llm.py               #   LLM(model_path).generate(...)（R3）
│       └── cli.py               #   qslab CLI（R3）
│
├── qslab/config.py              # 配置 dataclass（跨层共享，放顶层）
└── qslab/sampler.py             # 采样数学（L3 用，放 L3 或顶层）

adapters/                        # 唯一可 import transformers/vllm 的地方
├── tokenizer.py
└── model_loader.py              # （从 qslab/models/loader.py 拆出 HF 相关）

benchmarks/                      # 评测入口（保留，R4 重构）
tests/                          # pytest（R5）
scripts/                        # 只保留构建/环境脚本（R4 清理）
notes/ results/ docs/            # 保留
third_party/                     # cutlass/marlin（gitignore）
```

## 依赖规则（铁律，R1 落地并在 R5 用测试固化）

```
L0 ← L1 ← L2 ← L3 ← L4        （只允许相邻向下）
```

**唯一的例外**：`adapters/` 可被 L2/L3/L4 import（提供 tokenizer/HF 加载）。

**R1 必须消除的现有违规**：
- `qslab/model/w4linear.py` → `kernels.qslab_kernels.ops`（L2 直连 L0）
  → 改为经 L1 的 `quant.backends.get_backend(...)`
- `qslab/models/loader.py` → transformers（L2 直连外部 HF）
  → 拆出 HF 部分到 `adapters/model_loader.py`，L2 只接受已加载的 config/dict

## 迁移策略：只搬不改

R1 只做**文件移动 + import 路径修正**，不改任何逻辑。每个子步骤后跑一次 oracle 对齐（1.7B，快）+ 关键脚本冒烟，确认行为不变。

分批迁移顺序（每批一个 commit）：
```
R1-a  qslab/cache/ → qslab/engine/cache/          （含 strategy 预留）
R1-b  qslab/spec/  → qslab/engine/spec/           （verify.py → spec/engine.py）
R1-c  qslab/model/ → qslab/models/                （+ HF 部分拆到 adapters）
R1-d  quantizer/   → qslab/quant/
R1-e  kernels/qslab_kernels/ → qslab/kernels/
R1-f  qslab/engine.py → qslab/engine/core.py + 兼容 shim
R1-g  依赖规则检查脚本 + 修复 w4linear 跨层
```

## 兼容期策略

迁移期间保持**旧的 import 路径可用**（在旧位置留一行 re-export），避免所有 scripts/benchmarks 同时炸。R4 清理时统一删除 shim，并把脚本改为新路径。

## 验收（每批）

1. `python -c "from qslab.<new> import <Thing>"` 全部可导入
2. oracle 对齐脚本 PASS（1.7B W4，token 级一致）
3. 8B W4 冒烟：生成 32 token 成功且与基线文本一致
4. 无循环 import（`python -X importtime -c "import qslab"` 无报错）

## R2 预留（本阶段不动，只留位置）

- `qslab/quant/backends.py`：QuantBackend 抽象（R2 填）
- `qslab/engine/cache/strategy.py`：KVCacheStrategy（R2 填，**paged 实现位留给 M7**）
- `qslab/engine/spec/modes.py`：SpeculationMode（R2 填）
