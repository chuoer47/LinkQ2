from pathlib import Path

RT = Path("/home/<user>/<workdir>/qserve-lab/qslab/runtime")

# ---- 1. llm_engine.py: strip TP multiprocessing, fix imports ----
p = RT / "llm_engine.py"
s = p.read_text(encoding="utf-8")
s = s.replace("from nanovllm.config import Config", "from qslab.runtime.config import Config")
s = s.replace("from nanovllm.engine.sequence import Sequence, SequenceStatus",
              "from qslab.runtime.sequence import Sequence, SequenceStatus")
s = s.replace("from nanovllm.engine.model_runner import ModelRunner",
              "from qslab.runtime.model_runner import ModelRunner")
# strip the TP process spawn: qslab is single-GPU
old_tp = """        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)"""
new_tp = """        # qslab: single-GPU runtime; nano-vllm's TP worker processes removed.
        self.model_runner = ModelRunner(config, 0, self.events)"""
assert old_tp in s
s = s.replace(old_tp, new_tp)
s = s.replace("        self.ps = []\n        self.events = []\n", "")
# events list still referenced by ModelRunner; keep an empty list
s = s.replace("self.model_runner = ModelRunner(config, 0, self.events)",
              "self.events = []\n        self.model_runner = ModelRunner(config, 0, self.events)")
# exit(): no workers to join
s = s.replace("""    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()""",
"""    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner""")
p.write_text(s, encoding="utf-8")
print("llm_engine.py: TP stripped, imports fixed")

# ---- 2. model_runner.py: imports + no-TP paths ----
p = RT / "model_runner.py"
s = p.read_text(encoding="utf-8")
s = s.replace("from nanovllm.config import Config", "from qslab.runtime.config import Config")
s = s.replace("from nanovllm.engine.sequence import Sequence", "from qslab.runtime.sequence import Sequence")
s = s.replace("from nanovllm.models.qwen3 import Qwen3ForCausalLM", "from qslab.runtime.qwen3 import Qwen3ForCausalLM")
s = s.replace("from nanovllm.layers.sampler import Sampler", "from qslab.runtime.sampler import Sampler")
s = s.replace("from nanovllm.utils.context import set_context, get_context, reset_context",
              "from qslab.runtime.context import set_context, get_context, reset_context")
s = s.replace("from nanovllm.utils.loader import load_model", "from qslab.runtime.loader import load_model")
# strip dist init (single GPU)
old_dist = """        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)"""
assert old_dist in s
s = s.replace(old_dist, "        torch.cuda.set_device(rank)")
# strip the TP shm loop in __init__
s = s.replace("""        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

""", "")
s = s.replace("""    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:""",
"""    def exit(self):
        if not self.enforce_eager:""")
# loop/read_shm/write_shm/call: call is used by llm_engine; simplify call
s = s.replace("""    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)""",
"""    def call(self, method_name, *args):
        method = getattr(self, method_name, None)
        return method(*args)""")
p.write_text(s, encoding="utf-8")
print("model_runner.py: TP stripped, imports fixed")

# ---- 3. sequence.py / block_manager.py / scheduler.py imports ----
for name in ["sequence.py", "block_manager.py", "scheduler.py"]:
    p = RT / name
    s = p.read_text(encoding="utf-8")
    s = s.replace("from nanovllm.", "from qslab.runtime.")
    p.write_text(s, encoding="utf-8")
print("sequence/block_manager/scheduler: imports fixed")

# ---- 4. context.py stays ----
p = RT / "context.py"
s = p.read_text(encoding="utf-8")
s = s.replace("from nanovllm.", "from qslab.runtime.")
p.write_text(s, encoding="utf-8")
print("context.py: ok")

# ---- 5. config.py: block size 128, merge qslab semantics ----
p = RT / "config.py"
s = p.read_text(encoding="utf-8")
s = s.replace("from nanovllm.config import Config", "")
s = s.replace("    kvcache_block_size: int = 256", "    kvcache_block_size: int = 128")
s = s.replace("assert self.kvcache_block_size % 256 == 0",
              "assert self.kvcache_block_size % 128 == 0  # qslab: matches KV4 block alignment")
p.write_text(s, encoding="utf-8")
print("config.py: block size 128, assert fixed")
