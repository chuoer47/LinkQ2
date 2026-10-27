"""Use the same questions and timing boundaries for the speculative methods."""
import argparse, json
from pathlib import Path
from statistics import median
from time import perf_counter
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from qslab.api import LLM, SamplingParams
from benchmarks.common import PROMPTS, save, tuning

@torch.inference_mode()
def native_run(target, draft, tokenizer, ids, count):
    tensor = torch.tensor([ids], device='cuda')
    calls, first_ms = 0, None
    def after(_module, _args, _kwargs, _output):
        nonlocal calls, first_ms
        calls += 1
        if calls == 1:
            torch.cuda.synchronize()
            first_ms = (perf_counter()-started)*1000
    hook = target.register_forward_hook(after, with_kwargs=True)
    torch.cuda.synchronize()
    started = perf_counter()
    try:
        if draft is None:
            output = target.generate(tensor, do_sample=False, max_new_tokens=count,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id,
                temperature=None, top_p=None, top_k=None, use_cache=True)
        else:
            output = draft.spec_generate(target=target, input_ids=tensor, max_new_tokens=count,
                stop_token_ids=[tokenizer.eos_token_id], temperature=0.)
        torch.cuda.synchronize()
        total_ms = (perf_counter()-started)*1000
    finally:
        hook.remove()
    tokens = output[0,len(ids):].tolist()
    rounds = calls-1 if draft is not None else 0
    return {'total_ms': total_ms, 'first_phase_ms': first_ms,
        'decode_ms': total_ms-first_ms, 'output_tokens': len(tokens),
        'tpot_ms': (total_ms-first_ms)/max(len(tokens)-1,1),
        'verify_rounds': rounds, 'mean_len': (len(tokens)-1)/rounds if rounds else 0.,
        'target_calls': calls}

def runtime_run(llm,ids,count):
    engine=llm.engine;engine.scheduler.reset_spec_stats()
    torch.cuda.synchronize();started=perf_counter();first=None
    seq=engine.add_request(ids,SamplingParams(max_tokens=count,temperature=0,ignore_eos=False))
    while not engine.is_finished():
        engine.step()
        if first is None and seq.num_completion_tokens:
            torch.cuda.synchronize();first=(perf_counter()-started)*1000
    torch.cuda.synchronize();total=(perf_counter()-started)*1000
    n=seq.num_completion_tokens;stats=engine.scheduler.spec_stats
    return {'total_ms':total,'first_phase_ms':first,'output_tokens':n,
            'tpot_ms':(total-first)/max(1,n-1),'verify_rounds':stats['steps'],
            'mean_len':stats['mean_len'],
            'acceptance_rate':stats['acceptance_rate']}

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--method',choices=['baseline','ngram','draft','dflash'],required=True)
    p.add_argument('--model',default='models/Qwen3-8B-W4-rtn')
    p.add_argument('--kv',choices=['fp16','kv4'],default='kv4')
    p.add_argument('--native',action='store_true')
    p.add_argument('--draft-graph',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--window',type=int,default=4)
    p.add_argument('--max-tokens',type=int,default=256)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--limit',type=int,default=2)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(4);torch.manual_seed(123);tuning()
    tokenizer=AutoTokenizer.from_pretrained('models/Qwen3-8B',local_files_only=True)
    all_cases=json.loads(PROMPTS.read_text())['cases']
    # The small default includes one mathematical and one programming question.
    cases=([all_cases[0],all_cases[2]] if args.limit==2 else all_cases[:args.limit])
    ids=[tokenizer.apply_chat_template([{'role':'user','content':c['prompt']}],
            tokenize=True,add_generation_prompt=True,enable_thinking=False) for c in cases]
    llm=target=draft=None;window=0
    if args.native:
        if args.method not in ['baseline','dflash']:p.error('native supports baseline and dflash')
        target=AutoModelForCausalLM.from_pretrained('models/Qwen3-8B',torch_dtype=torch.bfloat16,
            attn_implementation='sdpa',local_files_only=True).cuda().eval()
        if args.method=='dflash':
            draft=AutoModel.from_pretrained('models/Qwen3-8B-DFlash-b16',torch_dtype=torch.bfloat16,
                trust_remote_code=True,attn_implementation='sdpa',local_files_only=True).cuda().eval()
        window=draft.config.block_size-1 if draft is not None else 0
        run=lambda tokens,count:native_run(target,draft,tokenizer,tokens,count)
    else:
        extra={}
        if args.method=='draft':
            extra=dict(draft='models/Qwen3-1.7B-W4-rtn',draft_kv_cache_dtype='kv4',
                draft_smooth_kv='datasets/kv4/qwen3-1.7b.pt',draft_num_kvcache_blocks=12)
        elif args.method=='dflash':extra=dict(draft='models/Qwen3-8B-DFlash-b16',dflash_graph=args.draft_graph)
        llm=LLM(args.model,kv_cache_dtype=args.kv,smooth_kv='datasets/kv4/qwen3-8b.pt',
            spec=None if args.method=='baseline' else args.method,spec_gamma=args.window,
            enforce_eager=False,enable_chunked_prefill=True,max_num_seqs=1,max_model_len=1024,
            max_num_batched_tokens=64,num_kvcache_blocks=8,**extra)
        llm.engine.scheduler.block_manager.ENABLE_PREFIX_CACHE=False
        window=llm.engine.scheduler.gamma
        run=lambda tokens,count:runtime_run(llm,tokens,count)
    rows=[]
    try:
        for prompt in ids:run(prompt,8)
        torch.cuda.reset_peak_memory_stats()
        for case,prompt in zip(cases,ids):
            samples=[run(prompt,args.max_tokens) for _ in range(args.repeats)]
            row={'case':case['id'],'input_tokens':len(prompt),
                 **{k:round(median(x[k] for x in samples),3) for k in
                    ['output_tokens','total_ms','first_phase_ms','tpot_ms','mean_len']},
                 'time_samples_ms':[round(x['total_ms'],3) for x in samples]}
            if not args.native:row['acceptance_rate']=round(median(x['acceptance_rate'] for x in samples),4)
            rows.append(row)
        result={'method':args.method,'model':Path(args.model).name,'kv':'bf16' if args.native else args.kv,
                'backend':'transformers' if args.native else 'runtime',
                'candidate_window':window,
                'peak_allocated_mib':round(torch.cuda.max_memory_allocated()/2**20,2),'measurements':rows}
    finally:
        if llm:llm.close()
    save(args.out,result);tuning(False)

if __name__=='__main__':main()
