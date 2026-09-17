import sys
sys.path.insert(0, "/home/<user>/<workdir>/qserve-lab")
from qslab.runtime.llm_engine import LLMEngine
from qslab.runtime.sampling_params import SamplingParams

MODEL = "/home/<user>/<workdir>/qserve-lab/models/Qwen3-1.7B"
engine = LLMEngine(model=MODEL, max_model_len=4096, max_num_seqs=8,
                   enforce_eager=True, gpu_memory_utilization=0.6)
engine.add_request(" The capital of France is",
                   SamplingParams(temperature=0.0, max_tokens=16))
seqs, is_prefill = engine.scheduler.schedule()
print("scheduled:", len(seqs), "is_prefill:", is_prefill)
for s in seqs:
    print("  seq:", s.seq_id, "scheduled:", s.num_scheduled_tokens)
print("running:", len(engine.scheduler.running), "waiting:", len(engine.scheduler.waiting))

# run one step and inspect logits/temps
from qslab.runtime.context import set_context, reset_context
input_ids, positions = engine.model_runner.prepare_prefill(seqs) if is_prefill else engine.model_runner.prepare_decode(seqs)
print("input_ids:", input_ids.shape, "positions:", positions.shape)
hidden = engine.model_runner.model(input_ids, positions)
print("hidden:", hidden.shape)
logits = engine.model_runner.model.compute_logits(hidden)
print("logits:", logits.shape)
temps = engine.model_runner.prepare_sample(seqs)
print("temps:", temps.shape, temps)
