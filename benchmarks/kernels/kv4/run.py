from benchmarks.common import publish
"""Small, model-free paired attention benchmark using the runtime path."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
from zoneinfo import ZoneInfo


def measure(call, torch, repeats):
    from triton.testing import do_bench_cudagraph
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    # Mature Triton timing helper excludes Python launch gaps and compilation.
    values = [do_bench_cudagraph(call, rep=30) * 1000 for _ in range(repeats)]
    return {'median_us': round(median(values),3), 'samples_us': [round(v,3) for v in values]}


def make_case(torch, batch, length, queries, prefill):
    from qslab.kernels.kv4 import store_kv_quant
    from qslab.kernels.fp16_kv import store_kvcache
    from qslab.runtime.model.attention import PagedAttention
    from qslab.runtime.model.context import set_context, get_context
    hq, hkv, d, page = 32, 8, 128, 256
    pages = (length + queries + page - 1) // page
    capacity = batch * pages * page
    # Reverse page order so correctness cannot depend on contiguous logical pages.
    table = torch.arange(batch * pages - 1, -1, -1, device='cuda', dtype=torch.int32).reshape(batch, pages)
    positions = torch.arange(length + queries, device='cuda')
    slots = (table[:, positions // page] * page + positions % page).flatten().int()
    raw = torch.randn(batch * (length + queries), hkv, d, device='cuda', dtype=torch.float16) * 0.5
    values = torch.randn_like(raw)
    scale = (raw.float().abs().amax(0) / 7).clamp_min(1e-4).half()
    lam = torch.rand(hkv, d, device='cuda', dtype=torch.float16) + 0.5
    fp_k = torch.zeros(batch * pages, page, hkv, d, device='cuda', dtype=torch.float16)
    fp_v = torch.zeros_like(fp_k)
    kq = torch.zeros(capacity, hkv, d // 8, device='cuda', dtype=torch.uint32)
    vq = torch.zeros_like(kq)
    vs = torch.zeros(capacity, hkv, d // 64, device='cuda', dtype=torch.float16)
    store_kvcache(raw, values, fp_k, fp_v, slots)
    store_kv_quant(raw, values, (kq, scale), (vq, vs), slots)
    # Inputs reproduce the strided slices of a fused QKV projection.
    fused = torch.randn(batch * queries, hq + hkv * 2, d, device='cuda', dtype=torch.float16) * 0.5
    q, k, v = fused.split([hq, hkv, hkv], dim=1)
    modules = {}
    for name in ['fp16', 'kv4']:
        attn = PagedAttention(hq, d, d ** -0.5, hkv, block_n=page)
        attn.kv_cache_dtype = name
        attn.set_smooth_lambda(lam)
        attn.k_cache, attn.v_cache = (fp_k, fp_v) if name == 'fp16' else ((kq, scale), (vq, vs))
        modules[name] = attn
    current = slots.reshape(batch, length + queries)[:, length:].flatten().contiguous()
    lens = torch.full((batch,), length + 1, device='cuda', dtype=torch.int32)
    if prefill:
        cuq = torch.arange(batch + 1, device='cuda', dtype=torch.int32) * queries
        cuk = torch.arange(batch + 1, device='cuda', dtype=torch.int32) * (length + queries)
        prefix = slots.reshape(batch, length + queries)[:, :length].flatten().contiguous()
        plan = [(i * length, length, i * queries, (i + 1) * queries, True) for i in range(batch)]
        set_context(True, cuq, cuk, queries, length + queries, current,
                    block_tables=table, prefix_slots=prefix, prefix_plan=plan)
    else:
        set_context(False, slot_mapping=current, context_lens=lens, block_tables=table, verify_m=queries)
    return modules, (q, k, v), get_context(), slots, raw, values


def stages(torch, attn, inputs, ctx):
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    from qslab.kernels.fp16_kv import store_kvcache
    from qslab.kernels.kv4 import materialize_kv, paged_attention_decode, store_kv_quant
    q, k, v = inputs
    def smooth():
        return q * attn.lam_q, k / attn.lam
    qs, ks = smooth()
    calls = {'smooth': smooth}
    if attn.kv_cache_dtype == 'fp16':
        calls['write'] = lambda: store_kvcache(ks, v, attn.k_cache, attn.v_cache, ctx.slot_mapping)
        if ctx.is_prefill:
            calls['attention'] = lambda: flash_attn_varlen_func(qs, attn.k_cache, attn.v_cache,
                cu_seqlens_q=ctx.cu_seqlens_q, cu_seqlens_k=ctx.cu_seqlens_k,
                max_seqlen_q=ctx.max_seqlen_q, max_seqlen_k=ctx.max_seqlen_k,
                causal=True, block_table=ctx.block_tables)
        else:
            b = ctx.context_lens.numel()
            calls['attention'] = lambda: flash_attn_with_kvcache(qs.reshape(b, ctx.verify_m, 32, 128),
                attn.k_cache, attn.v_cache, cache_seqlens=ctx.context_lens + ctx.verify_m - 1,
                block_table=ctx.block_tables, causal=True).flatten(0, 1)
    else:
        if not v.is_contiguous():
            calls['contiguous_copy_reference'] = lambda: (ks.contiguous(), v.contiguous())
        calls['write'] = lambda: store_kv_quant(ks, v, attn.k_cache, attn.v_cache, ctx.slot_mapping)
        if ctx.is_prefill:
            calls['materialize'] = lambda: materialize_kv(attn.k_cache, attn.v_cache, ctx.prefix_slots)
            calls['prefix_with_concat'] = lambda: attn._materialize_prefix(ctx, ks, v)
            kp, vp = calls['prefix_with_concat']()
            calls['attention'] = lambda: flash_attn_varlen_func(qs, kp, vp,
                cu_seqlens_q=ctx.cu_seqlens_q, cu_seqlens_k=ctx.cu_seqlens_k,
                max_seqlen_q=ctx.max_seqlen_q, max_seqlen_k=ctx.max_seqlen_k, causal=True)
        else:
            calls['attention'] = lambda: paged_attention_decode(qs, attn.k_cache, attn.v_cache,
                ctx.block_tables, ctx.context_lens, block_n=256, num_kv_heads=8, verify_m=ctx.verify_m)
    calls['layer'] = lambda: attn(*inputs)
    return calls


def accuracy(torch, attn, inputs, ctx, output):
    from qslab.kernels.kv4 import materialize_kv
    # Reference uses the exact quantized values read by this attention path.
    q, k, v = inputs
    q = (q * attn.lam_q).float()
    ks = k / attn.lam
    b = ctx.block_tables.shape[0]
    m = q.shape[0] // b
    refs = []
    heads = torch.arange(32, device='cuda') // 4
    for i in range(b):
        length = ctx.max_seqlen_k if ctx.is_prefill else int(ctx.context_lens[i]) + m - 1
        pos = torch.arange(length, device='cuda')
        slots = ctx.block_tables[i, pos // 256].long() * 256 + pos % 256
        if attn.kv_cache_dtype == 'kv4':
            kr, vr = materialize_kv(attn.k_cache, attn.v_cache, slots)
        else:
            kr, vr = attn.k_cache.flatten(0, 1)[slots], attn.v_cache.flatten(0, 1)[slots]
        if ctx.is_prefill and attn.kv_cache_dtype == 'kv4':
            kr[-m:], vr[-m:] = ks[i*m:(i+1)*m], v[i*m:(i+1)*m]
        for row in range(m):
            n = length - m + row + 1
            scores = torch.einsum('hd,thd->ht', q[i*m+row], kr[:n, heads].float()) * 128**-0.5
            refs.append(torch.einsum('ht,thd->hd', scores.softmax(-1), vr[:n, heads].float()))
    ref = torch.stack(refs)
    gap = (output.float() - ref).abs()
    return {'max_abs': round(gap.max().item(),6), 'relative_l2': round(((output.float()-ref).norm()/ref.norm()).item(),6)}


def report(out, rows):
    lines = ['# FP16 / KV4 Attention 分解', '',
        'GPU Graph 重放计时，单位微秒，数值为重复测量中位数。layer 为真实 PagedAttention.forward。', '',
        '| 场景 B/L/M | FP16 layer | KV4 layer | FP16/KV4 | FP16 attention | KV4 attention | KV4 前缀物化+拼接 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for case in rows:
        f, q = case['modes']['fp16']['stages'], case['modes']['kv4']['stages']
        us = lambda d,k: d.get(k,{}).get('median_us',0)
        lines.append(f"| {case['kind']} {case['batch']}/{case['length']}/{case['queries']} | {us(f,'layer'):.2f} | {us(q,'layer'):.2f} | {us(f,'layer')/us(q,'layer'):.3f}× | {us(f,'attention'):.2f} | {us(q,'attention'):.2f} | {us(q,'prefix_with_concat'):.2f} |")
    lines += ['', '分项独立计时，不能精确相加解释 layer；前缀物化包含在 prefix_with_concat 内，不能重复求和。',
        '数值正确性针对各自实际缓存值；量化与 FP16 的算法差异另存 JSON。容量相同，KV 字节包含 scale，不包含临时工作区。']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',default='1')
    parser.add_argument('--contexts',nargs='+',type=int,default=[128,512,2048])
    parser.add_argument('--batches',nargs='+',type=int,default=[1,4])
    parser.add_argument('--queries',nargs='+',type=int,default=[1,16])
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    import torch
    from qslab.runtime.model.context import reset_context
    torch.set_num_threads(4)
    torch.manual_seed(123)
    out=args.out or Path('results/kernels/kv4')
    out.mkdir(parents=True,exist_ok=True)
    cases=[(b,l,m,False) for b in args.batches for l in args.contexts for m in args.queries]
    cases += [(1,l,64,True) for l in args.contexts]
    rows=[]
    with torch.inference_mode():
        for b,l,m,prefill in cases:
            modules,inputs,ctx,*_=make_case(torch,b,l,m,prefill)
            row={'kind':'chunk' if prefill else 'decode' if m==1 else 'verify',
                 'batch':b,'length':l,'queries':m,'modes':{}}
            outputs={}
            for mode,attn in modules.items():
                calls=stages(torch,attn,inputs,ctx)
                outputs[mode]=calls['layer']().clone()
                error=accuracy(torch,attn,inputs,ctx,outputs[mode])
                assert error['max_abs']<0.004, (mode,row,error)
                metrics={key:measure(call,torch,args.repeats) for key,call in calls.items()}
                tensors=[attn.k_cache,attn.v_cache] if mode=='fp16' else [*attn.k_cache,*attn.v_cache]
                row['modes'][mode]={'stages':metrics,'reference_error':error,
                    'kv_bytes':sum(t.numel()*t.element_size() for t in tensors)}
            row['kv4_vs_fp16_relative_l2']=round(((outputs['kv4'].float()-outputs['fp16'].float()).norm()/outputs['fp16'].float().norm()).item(),6)
            rows.append(row)
            (out/'results.json').write_text(json.dumps(rows,indent=2))
            print(row['kind'],b,l,m,{k:round(v['stages']['layer']['median_us'],2) for k,v in row['modes'].items()},flush=True)
            reset_context()
            del modules,inputs,ctx,calls,outputs
    report(out,rows)
    print(out/'report.md')

    publish(out,Path(__file__).parent)


if __name__=='__main__':main()
