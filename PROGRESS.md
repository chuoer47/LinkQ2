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
- [ ] QuantBackend / KVCacheStrategy(预留 paged) / SpeculationMode + 注册表

### R3 统一入口
- [ ] LLM(model_path).generate(...) + qslab CLI

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
