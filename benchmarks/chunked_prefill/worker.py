"""Drive the public engine step path with three deterministic request arrivals."""
import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from qslab.api import LLM
from qslab.runtime.sampling_params import SamplingParams
from benchmarks.engine.worker import released


def workload(engine, ids):
    jobs, stamps, submitted, trace = [], {}, {}, []
    original = engine.scheduler.schedule
    started = perf_counter()

    def submit(prompt, tokens):
        seq = engine.add_request(prompt, SamplingParams(max_tokens=tokens, temperature=0, ignore_eos=True))
        jobs.append(seq)
        submitted[seq.seq_id] = perf_counter()
        stamps[seq.seq_id] = []

    def scheduled():
        eligible = {s.seq_id for s in engine.scheduler.running}
        seqs, has_prefill = original()
        trace.append({'has_prefill': has_prefill, 'ids': [s.seq_id for s in seqs],
            'tokens': [s.num_scheduled_tokens for s in seqs],
            'decode_ids': [s.seq_id for s in seqs if not s.is_prefill],
            'sample': [s.should_sample for s in seqs],
            'deferred_decode_ids': sorted(eligible - {s.seq_id for s in seqs})})
        return seqs, has_prefill

    engine.scheduler.schedule = scheduled
    initial_replays = engine.model_runner.graph_replays
    try:
        submit(ids[:32], 16)
        step = 0
        while not engine.is_finished():
            if step == 2:
                submit(ids[:512], 8)
                submit(ids[128:160], 8)
            before = {s.seq_id: s.num_completion_tokens for s in jobs}
            engine.step()
            torch.cuda.synchronize()
            now = perf_counter()
            for seq in jobs:
                if seq.num_completion_tokens > before[seq.seq_id]:
                    stamps[seq.seq_id].append(now)
            step += 1
            if step > 100:
                raise RuntimeError('small workload did not finish within 100 steps')
        elapsed = perf_counter() - started
    finally:
        del engine.scheduler.schedule
    assert len(jobs) == 3 and [s.num_completion_tokens for s in jobs] == [16, 8, 8]
    assert released(engine)
    assert all(sum(t['tokens']) <= engine.scheduler.max_num_batched_tokens for t in trace)
    requests = []
    for seq in jobs:
        times = stamps[seq.seq_id]
        intervals = [(b-a)*1000 for a,b in zip(times,times[1:])]
        requests.append({'prompt_tokens': seq.num_prompt_tokens, 'output_tokens': seq.num_completion_tokens,
            'ttft_ms': (times[0]-submitted[seq.seq_id])*1000,
            'tpot_ms': sum(intervals)/len(intervals), 'max_token_gap_ms': max(intervals),
            'e2e_ms': (times[-1]-submitted[seq.seq_id])*1000, 'token_ids': seq.completion_token_ids})
    return {'total_ms': elapsed*1000, 'output_tokens_per_s':32/elapsed,
        'requests':requests, 'steps':step,
        'mixed_steps':sum(t['has_prefill'] and bool(t['decode_ids']) for t in trace),
        'short_decode_deferred_steps':sum(jobs[0].seq_id in t['deferred_decode_ids'] for t in trace),
        'graph_replays':engine.model_runner.graph_replays-initial_replays,
        'all_pages_returned':released(engine), 'trace':trace}


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--kv',choices=['fp16','kv4'],required=True)
    p.add_argument('--mixed',action='store_true')
    p.add_argument('--samples',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(123)
    from qslab.kernels.gemlite.core import load_config,save_config
    tuning=args.out.parent/'autotune.json'
    if tuning.exists():load_config(tuning)
    ids=json.loads(args.samples.read_text())['samples'][0]['token_ids']
    warm=torch.zeros(16,16,device='cuda',dtype=torch.float16)
    warm=warm@warm
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):product=warm@warm
    del warm,graph,product
    torch.cuda.synchronize();torch.cuda.empty_cache()
    library=torch.cuda.memory_allocated()
    llm=LLM(args.model,kv_cache_dtype=args.kv,
        smooth_kv='datasets/kv4/qwen3-8b.pt' if args.kv=='kv4' else None,
        max_model_len=768,max_num_seqs=4,max_num_batched_tokens=64,
        num_kvcache_blocks=8,kvcache_block_size=256,enforce_eager=False,
        enable_chunked_prefill=args.mixed)
    engine=llm.engine
    engine.scheduler.block_manager.ENABLE_PREFIX_CACHE=False
    resident=torch.cuda.memory_allocated()
    try:
        workload(engine,ids)
        torch.cuda.reset_peak_memory_stats()
        runs=[workload(engine,ids) for _ in range(3)]
        result={'model':args.model,'kv':args.kv,'mixed':args.mixed,'gpu':torch.cuda.get_device_name(),
            'resident_allocated_bytes':resident,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'library_baseline_allocated_bytes':library,'kv_bytes':llm.kv_memory_bytes(),'runs':runs,
            'total_ms':median(r['total_ms'] for r in runs),
            'output_tokens_per_s':median(r['output_tokens_per_s'] for r in runs),
            'short_max_token_gap_ms':median(r['requests'][0]['max_token_gap_ms'] for r in runs),
            'short_tpot_ms':median(r['requests'][0]['tpot_ms'] for r in runs),
            'long_ttft_ms':median(r['requests'][1]['ttft_ms'] for r in runs)}
        assert all(r['all_pages_returned'] and r['graph_replays']>0 for r in runs)
        if args.mixed:
            assert all(r['mixed_steps']>0 and r['short_decode_deferred_steps']==0 for r in runs)
        else:
            assert all(r['mixed_steps']==0 and r['short_decode_deferred_steps']>0 for r in runs)
    finally:
        llm.close()
    result['allocated_bytes_after_close']=torch.cuda.memory_allocated()
    assert result['allocated_bytes_after_close']<=library+1024*1024
    result['passed']=True
    args.out.write_text(json.dumps(result,indent=2))
    save_config(tuning)


if __name__=='__main__':main()
