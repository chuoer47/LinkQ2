from pathlib import Path

p = Path("/home/<user>/<workdir>/qserve-lab/qslab/engine/core.py")
s = p.read_text()

# --- replace the inline cache-class dispatch with the L1 strategy factory ---
old_start = s.index("        # per-layer cache classes (M2): default all fp16, plan overrides")
old_end = s.index("        self.model.eval()")
old_block = s[old_start:old_end]

new_block = '''        # per-layer caches come from an L1 strategy (R2): the kv_mode /
        # plan-file decision lives in qslab.quant.kv_strategies, not here.
        from qslab.quant.kv_strategies_impl import build_strategy
        self.kv_strategy = build_strategy(kv_mode, kv_plan_path)
        self.kv_caches = patch_model(
            self.model,
            num_layers=self.model_cfg.num_hidden_layers,
            num_kv_heads=self.model_cfg.num_key_value_heads,
            head_dim=self.model_cfg.head_dim,
            max_len=self.model_cfg.max_position_embeddings,
            device=self.device,
            cache_factory=lambda i: self.kv_strategy.build(
                layer_idx=i,
                batch=1,
                num_kv_heads=self.model_cfg.num_key_value_heads,
                head_dim=self.model_cfg.head_dim,
                max_len=self.model_cfg.max_position_embeddings,
                device=self.device,
            ),
        )
'''
s = s[:old_start] + new_block + s[old_end:]
p.write_text(s)
print("engine/core.py: cache dispatch -> strategy factory")
