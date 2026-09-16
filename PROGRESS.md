# PROGRESS.md — qserve-lab 唯一状态源

## 当前阶段：**R0-R6 工程化重构**

### R0 flash-attn 安装
- [x] flash-attn 2.7.4 (cu124/torch2.5/py3.11 预编译 wheel) 装入 qslab env
      来源：mjun0812/flash-attention-prebuild-wheels v0.3.18
      验证：flash_attn_func 在 sm_89 上实测通过（2026-09-16）
      坑：wheel 文件名必须合规（fa.whl → 全名重命名），pip 才接受

### R1 分层骨架（待开工）
- [ ] L0 kernels / L1 quant / L2 models / L3 engine / L4 api
- [ ] 依赖规则：禁止跨层（w4linear 直连 kernels 的违规修复）

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
