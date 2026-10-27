"""Measure latency of three staggered requests with and without mixed prefill."""
import argparse, json
from pathlib import Path
from statistics import median
from time import perf_counter
import torch
from qslab.api import LLM, SamplingParams
from benchmarks.common import SAMPLES, save, tuning

def workload(engine,ids):
    jobs=[];arrivals={};deliveries={}
    def add(prompt,tokens):
        seq=engine.add_request(prompt,SamplingParams(max_tokens=tokens,ignore_eos=True))
        jobs.append(seq);arrivals[seq.seq_id]=perf_counter();deliveries[seq.seq_id]=[]
    torch.cuda.synchronize();started=perf_counter();add(ids[:32],16);step=0
    while not engine.is_finished():
        if step==2:add(ids[:512],8);add(ids[:32],8)
        before=[s.num_completion_tokens for s in jobs]
        engine.step();torch.cuda.synchronize();now=perf_counter()
        for seq,count in zip(jobs,before):
            if seq.num_completion_tokens>count:deliveries[seq.seq_id].append(now)
        step+=1
    elapsed=(perf_counter()-started)*1000;rows=[]
    for seq in jobs:
        ts=deliveries[seq.seq_id]
        rows.append({'input_tokens':seq.num_prompt_tokens,'output_tokens':seq.num_completion_tokens,
            'ttft_ms':round((ts[0]-arrivals[seq.seq_id])*1000,3),
            'tpot_ms':round((ts[-1]-ts[0])*1000/(len(ts)-1),3),
            'max_token_gap_ms':round(max(b-a for a,b in zip(ts,ts[1:]))*1000,3)})
    return {'e2e_ms':round(elapsed,3),'output_tokens_per_s':round(32000/elapsed,3),'requests':rows}

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--quantized',action='store_true')
    p.add_argument('--mixed',action='store_true')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(4);torch.manual_seed(123);tuning()
    model='models/Qwen3-8B-W4-rtn' if args.quantized else 'models/Qwen3-8B'
    with LLM(model,kv_cache_dtype='kv4' if args.quantized else 'fp16',
             smooth_kv='datasets/kv4/qwen3-8b.pt' if args.quantized else None,
             enforce_eager=False,enable_chunked_prefill=args.mixed,max_num_seqs=4,
             max_model_len=768,max_num_batched_tokens=64,num_kvcache_blocks=8) as llm:
        engine=llm.engine;engine.scheduler.block_manager.ENABLE_PREFIX_CACHE=False
        ids=json.loads(SAMPLES.read_text())['samples'][0]['token_ids']
        workload(engine,ids);rows=[workload(engine,ids) for _ in range(3)]
    requests=[{key:round(median(x['requests'][i][key] for x in rows),3)
               for key in rows[0]['requests'][i]} for i in range(3)]
    save(args.out,{'configuration':'w4-kv4' if args.quantized else 'fp16-fp16','mixed':args.mixed,
        'e2e_ms':median(r['e2e_ms'] for r in rows),'output_tokens_per_s':median(r['output_tokens_per_s'] for r in rows),
        'e2e_samples_ms':[r['e2e_ms'] for r in rows],'requests':requests});tuning(False)

if __name__=='__main__':main()
