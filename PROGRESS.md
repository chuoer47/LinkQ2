# PROGRESS.md — qserve-lab 唯一状态源

## 当前阶段：**R0-R6 工程化重构**

### R0 flash-attn 安装
- [x] flash-attn 2.7.4 (cu124/torch2.5/py3.11 预编译 wheel) 装入 qslab env
      来源：mjun0812/flash-attention-prebuild-wheels v0.3.18
      验证：flash_attn_func 在 sm_89 上实测通过（2026-09-16）
      坑：wheel 文件名必须合规（fa.whl → 全名重命名），pip 才接受

### R1 分层骨架
- [x] L0 qslab/kernels（csrc+ops+marlin）/ L1 qslab/quant（算法+packfmt+cache）/ L2 qslab/models / L3 qslab/engine（core+cache→quant+spec）/ L4 qslab/api（空壳待 R3）
- [x] 设计修正：KV cache 归属 L1（"数据怎么存"）而非 L3——解决了 L2↔L3 循环依赖（2026-09-16）
- [x] transformers 违规修复：loader 的 HF 构建移到 adapters/model_builder.py，L2 零 transformers（2026-09-16）
- [ ] 遗留：L2 w4linear 直连 L0 kernels（3 处）——R2 的 QuantBackend 正式消除
- [x] 验收：1.7B oracle 冒烟 PASS（输出与重构前一致）

### R2 策略接口
- [x] registry.py 通用注册表（装饰器注册 + 工厂查询）
- [x] QuantBackend(L1)：w4.v1 / w4.marlin / w4.auto 三后端，**M 分派内化到 backend**；W4Linear 只调 backend.linear()，L2->L0 直连彻底消除（2026-09-16）
- [x] KVCacheStrategy(L1)：fp16/kv8/kv4/kv4.plan 策略 + 注册表；engine 的内联 kv_mode 分派改为策略工厂；**paged 位已预留**（M7 用）
- [x] SpeculationMode(L3)：chained/lookahead/dynamic 策略类，验证/回滚逻辑与提案策略解耦；补回 lookahead 空提案回退（重构中丢失的路径）（2026-09-16）
- [x] 验收：三模式 lossless=True（ar 4.00/1.75/4.71）；QuantBackend 冒烟输出与基线一致

### R3 统一入口
- [x] qslab/api/llm.py：LLM 门面（model + w4 + kv_mode + draft 正交组合，SamplingParams，stats/kv_memory/weight_memory 查询）（2026-09-16）
- [x] qslab/api/cli.py：python -m qslab.api.cli generate --model ... [--w4 --kv-mode --draft --spec-mode]
- [x] 修复 R2 引入的真 bug：lookahead 空提案回退分支的 commit 位置算错（start_pos 少 1）导致 KV 错位、无损性失败——**由统一入口的端到端测试抓出**（2026-09-16）
- [x] 验收：三种 CLI 组合全部输出正确；W4+lookahead LOSSLESS=True

### R4 仓库清理
- [ ] dbg_* 归档、m*_ 重构、committed 二进制与 results 日志出 git

### R5 测试与锁定
- [ ] pytest 套件 + environment.yml + requirements.txt

### R6 文档
- [ ] README + 架构文档（含 nano-vllm 整合与 M7 量化 paged attention 路线图）

### 行为无回归基线（每阶段复测）
8B W4 e2e 39.8 tok/s | PPL 17.31 | KV4 省 3.5× | lookahead 1.44× | oracle 对齐 PASS

### R0 断点
（无——R1 待开工）
