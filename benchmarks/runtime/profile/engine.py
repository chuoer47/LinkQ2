"""Capture a warmed runtime request for Nsight."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--kv',choices=['fp16','kv4'],required=True)
p.add_argument('--context',type=int,default=512)
p.add_argument('--tokens',type=int,default=16)
args=p.parse_args()
root=Path.cwd()
import torch
from qslab.api import LLM,SamplingParams

torch.set_num_threads(4)
torch.manual_seed(123)
from benchmarks.common import tuning
tuning()
samples=json.loads((root/'benchmarks/data/samples.json').read_text())['samples']
ids=[t for sample in samples for t in sample['token_ids']][:args.context]
assert len(ids)==args.context
with torch.inference_mode():
    llm=LLM(root/'models/Qwen3-8B-W4-rtn',kv_cache_dtype=args.kv,
        smooth_kv=str(root/'datasets/kv4/qwen3-8b.pt'),enforce_eager=False,
        enable_chunked_prefill=True,max_num_seqs=1,max_model_len=4096,
        max_num_batched_tokens=64,num_kvcache_blocks=16)
    engine=llm.engine
    engine.scheduler.block_manager.ENABLE_PREFIX_CACHE=False
    params=SamplingParams(max_tokens=args.tokens,temperature=0,ignore_eos=True)
    # Keep both traces on the same fixed continuation as the paired benchmark.
    handle=engine.model_runner.sampler.register_forward_hook(lambda _m,_a,o:torch.full_like(o,42))
    for _ in range(2):engine.generate([ids],params,use_tqdm=False)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    seq=engine.add_request(ids,params)
    step=0
    while not seq.is_finished:
        phase='prefill' if seq.num_completion_tokens==0 else 'decode'
        torch.cuda.nvtx.range_push(f'engine/{args.kv}/{phase}/step{step}')
        engine.step()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        step+=1
    torch.cuda.cudart().cudaProfilerStop()
    handle.remove()
    llm.close()
print('engine completed',args.kv,flush=True)
