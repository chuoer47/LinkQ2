import sys, torch
sys.path.insert(0, "/home/<user>/<workdir>/qserve-lab")
from qslab.config import EngineConfig
from qslab.engine import QslabEngine
from adapters.tokenizer import QwenTokenizerAdapter
tok = QwenTokenizerAdapter("/home/<user>/<workdir>/qserve-lab/models/Qwen3-1.7B")
ids = tok.encode(" The capital of France is")
cfg = EngineConfig(model_path="/home/<user>/<workdir>/qserve-lab/models/Qwen3-1.7B", device="cuda:0", max_new_tokens=24)
for mode in ["fp16", "kv4"]:
    eng = QslabEngine(cfg, kv_mode=mode, kv_plan_path="/home/<user>/<workdir>/qserve-lab/results/kv4_plan.json")
    print(mode, "gen:", tok.decode(eng.generate(ids, max_new_tokens=24))[:45], flush=True)
    del eng; torch.cuda.empty_cache()
