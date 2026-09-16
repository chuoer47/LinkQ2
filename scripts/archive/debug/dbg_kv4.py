import sys, torch
sys.path.insert(0, "/home/<user>/<workdir>/qserve-lab")
from qslab.config import EngineConfig
from qslab.engine import QslabEngine

cfg = EngineConfig(model_path="/home/<user>/<workdir>/qserve-lab/models/Qwen3-1.7B",
                   device="cuda:0", max_new_tokens=24)
eng = QslabEngine(cfg, kv_mode="kv4",
                  kv_plan_path="/home/<user>/<workdir>/qserve-lab/results/kv4_plan.json")
try:
    eng.generate([1, 2, 3, 4, 5], max_new_tokens=3)
    print("GEN OK")
except RuntimeError:
    import traceback
    tb = traceback.format_exc().splitlines()
    for line in tb[-10:]:
        print(line)
