"""Measure tensor peaks from model loading through generation."""
import argparse, json
from pathlib import Path
from time import perf_counter
import torch
from qslab.api import LLM, SamplingParams
from benchmarks.common import SAMPLES, save, tuning


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--kv',choices=['fp16','kv4'],required=True)
    p.add_argument('--context',type=int,required=True)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(4);torch.manual_seed(123);tuning()
    torch.cuda.init();peak_allocated=peak_reserved=0
    original_reset=torch.cuda.reset_peak_memory_stats
    def remember():
        nonlocal peak_allocated,peak_reserved
        peak_allocated=max(peak_allocated,torch.cuda.max_memory_allocated())
        peak_reserved=max(peak_reserved,torch.cuda.max_memory_reserved())
    def observed_reset(*a,**kw):
        # Model warmup resets its counters; retain the loading peak first.
        remember();return original_reset(*a,**kw)
    torch.cuda.reset_peak_memory_stats=observed_reset
    samples=json.loads(SAMPLES.read_text())['samples']
    ids=[t for sample in samples for t in sample['token_ids']][:args.context]
    assert len(ids)==args.context
    llm=None;rows=[]
    try:
        with torch.inference_mode():
            llm=LLM('models/Qwen3-8B-W4-rtn',kv_cache_dtype=args.kv,
                smooth_kv='datasets/kv4/qwen3-8b.pt',enforce_eager=False,
                enable_chunked_prefill=True,max_num_seqs=1,max_model_len=4096,
                max_num_batched_tokens=64,num_kvcache_blocks=16)
            engine=llm.engine;engine.scheduler.block_manager.ENABLE_PREFIX_CACHE=False
            # Identical continuations make the cache workload comparable.
            handle=engine.model_runner.sampler.register_forward_hook(lambda _m,_a,o:torch.full_like(o,42))
            try:
                for repeat in range(args.repeats+1):
                    torch.cuda.synchronize();started=perf_counter();first=None
                    seq=engine.add_request(ids,SamplingParams(max_tokens=32,temperature=0,ignore_eos=True))
                    while not seq.is_finished:
                        engine.step();torch.cuda.synchronize()
                        if first is None and seq.num_completion_tokens:first=perf_counter()
                    ended=perf_counter();remember()
                    assert seq.num_completion_tokens==32
                    assert not engine.scheduler.block_manager.used_block_ids
                    rows.append({'warmup':repeat==0,'tpot_ms':(ended-first)*1000/31})
            finally:handle.remove()
            result={'kv_bytes':llm.kv_memory_bytes(),'requests':rows,
                'lifetime_peak_allocated_bytes':peak_allocated,
                'lifetime_peak_reserved_bytes':peak_reserved}
    finally:
        if llm:llm.close()
        torch.cuda.reset_peak_memory_stats=original_reset
    save(args.out/'worker.json',result);tuning(False)

if __name__=='__main__':main()
