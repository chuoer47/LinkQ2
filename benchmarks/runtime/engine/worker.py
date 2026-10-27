"""Time requests through the scheduler, model, cache and sampler."""
import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter
import torch
from qslab.api import LLM
from qslab.runtime.sampling_params import SamplingParams
from benchmarks.common import SAMPLES, save, tuning
def released(engine):
    manager = engine.scheduler.block_manager
    return (engine.is_finished() and not manager.used_block_ids
            and len(manager.free_block_ids) == len(manager.blocks)
            and all(block.ref_count == 0 for block in manager.blocks))

def cycle(engine, ids, tokens):
    torch.cuda.synchronize()
    started = perf_counter()
    seq = engine.add_request(ids, SamplingParams(max_tokens=tokens, temperature=0, ignore_eos=True))
    first = last = None
    while not seq.is_finished:
        engine.step()
        torch.cuda.synchronize()
        if seq.num_completion_tokens:
            last = perf_counter()
            if first is None:
                first = last
    assert seq.num_completion_tokens == tokens and released(engine)
    total = last - started
    return {'ttft_ms': (first - started) * 1000,
            'tpot_ms': (last - first) * 1000 / (tokens - 1),
            'e2e_ms': total * 1000, 'output_tokens_per_s': tokens / total}

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--kv',choices=['fp16','kv4'],required=True)
    p.add_argument('--graph',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--chunked',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--calibration',default='datasets/kv4/qwen3-8b.pt')
    p.add_argument('--contexts',nargs='+',type=int,default=[128,512])
    p.add_argument('--tokens',type=int,default=32)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(4);torch.manual_seed(123);tuning()
    ids=json.loads(SAMPLES.read_text())['samples'][0]['token_ids']
    if any(length<=0 or length>len(ids) for length in args.contexts):
        p.error('输入长度须在固定样本的 1 到 '+str(len(ids))+' token 内')
    llm=LLM(args.model,kv_cache_dtype=args.kv,smooth_kv=args.calibration if args.kv=='kv4' else None,
        enforce_eager=not args.graph,enable_chunked_prefill=args.chunked,
        max_num_seqs=1,max_model_len=1024,max_num_batched_tokens=64 if args.chunked else 1024,
        num_kvcache_blocks=4)
    engine=llm.engine;engine.scheduler.block_manager.ENABLE_PREFIX_CACHE=False
    rows=[]
    try:
        for length in args.contexts:
            cycle(engine,ids[:length],args.tokens)
            times=[cycle(engine,ids[:length],args.tokens) for _ in range(args.repeats)]
            rows.append({'input_tokens':length,'output_tokens':args.tokens,
                **{k:round(median(row[k] for row in times),3) for k in times[0]},
                'e2e_samples_ms':[round(row['e2e_ms'],3) for row in times]})
        result={'model':Path(args.model).name,'kv':args.kv,'graph':args.graph,'chunked_prefill':args.chunked,
                'weight_mib':round(llm.weight_memory_bytes()/2**20,2),'kv_mib':round(llm.kv_memory_bytes()/2**20,2),
                'measurements':rows}
    finally:llm.close()
    save(args.out,result);tuning(False)

if __name__=='__main__':main()
