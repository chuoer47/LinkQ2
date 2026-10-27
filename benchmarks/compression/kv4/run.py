from benchmarks.common import publish
"""Compare K/V distributions and quantization error on real model activations."""
import argparse
import json
import os
from pathlib import Path
from benchmarks.common import SAMPLES, save, table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default='1')
    ap.add_argument('--model', default='models/Qwen3-8B')
    ap.add_argument('--calibration', default='datasets/w4a16/checkpoint_calibration.pt')
    ap.add_argument('--samples', default=str(SAMPLES))
    ap.add_argument('--n-seqs', type=int, default=2)
    ap.add_argument('--seq-len', type=int, default=256)
    ap.add_argument('--out', type=Path, default=Path('results/compression/kv4'))
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from adapters.model_builder import build_hf_causal_lm
    from scripts.build_smooth_kv import apply_rope_halfsplit
    torch.set_num_threads(4)
    torch.manual_seed(123)
    args.out.mkdir(parents=True, exist_ok=True)
    calibration = torch.load(args.calibration, map_location='cpu', weights_only=True)['token_ids']
    samples = json.loads(Path(args.samples).read_text())['samples'][:args.n_seqs]
    evaluation = [s['token_ids'][:args.seq_len] for s in samples]
    model = build_hf_causal_lm(args.model)
    cfg = model.config
    nl, hk, hq, d = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.num_attention_heads, cfg.head_dim
    captured = {}
    handles = []
    for li, layer in enumerate(model.model.layers):
        def hook(_m, a, kw, _out, li=li):
            captured[li] = (a[0] if a else kw['hidden_states']).detach()
        handles.append(layer.self_attn.register_forward_hook(hook, with_kwargs=True))
    def forward(ids):
        captured.clear()
        tokens = torch.tensor([ids],device='cuda')
        model.model(input_ids=tokens, use_cache=False)
        emb = model.model.embed_tokens(tokens)
        cos, sin = model.model.rotary_emb(emb, torch.arange(len(ids),device='cuda')[None])
        return cos[0], sin[0]
    def tensors(li, cos, sin, query=False):
        attn = model.model.layers[li].self_attn
        x = captured[li]
        k = attn.k_norm(attn.k_proj(x).view(-1,hk,d)).float()
        k = apply_rope_halfsplit(k,cos,sin)
        v = attn.v_proj(x).view(-1,hk,d).float()
        q = None
        if query:
            q = attn.q_norm(attn.q_proj(x).view(-1,hq,d)).float()
            q = apply_rope_halfsplit(q,cos,sin)
        return q,k,v
    kmax = torch.zeros(nl,hk,d,dtype=torch.float64)
    vmax = torch.zeros_like(kmax)
    with torch.inference_mode():
        for index,ids in enumerate(calibration):
            cos,sin = forward(ids)
            for li in range(nl):
                _,k,v=tensors(li,cos,sin)
                kmax[li]=torch.maximum(kmax[li],k.abs().amax(0).double().cpu())
                vmax[li]=torch.maximum(vmax[li],v.abs().amax(0).double().cpu())
            print('calibration',index+1,len(calibration),flush=True)
        half=d//2
        pair=torch.maximum(kmax[...,:half],kmax[...,half:]).clamp_min(1e-6).sqrt()
        lam=torch.cat([pair,pair],-1).half().float().cuda()
        # Use the same pre-FP16 lambda in the offline scale calculation as build_smooth_kv.
        lam_full=torch.cat([pair,pair],-1)
        sk=((kmax/lam_full).clamp_min(1e-4)/7).half().float().cuda()
        sk_raw=(kmax.clamp_min(1e-4)/7).half().float().cuda()
        sv_static=(vmax.clamp_min(1e-4)/7).half().float().cuda()
        def quant(x,s):
            u=x/s
            q=torch.floor(u.clamp(-8,7)+0.5)
            return q*s, ((u < -8)|(u > 7)).sum().item(), (q==0).sum().item()
        def dynamic(x,g):
            shaped=x.reshape(x.shape[0],hk,d//g,g)
            scale=(shaped.abs().amax(-1,keepdim=True)/7).clamp_min(1e-4).half().float()
            restored,clips,zeros=quant(shaped,scale)
            return restored.reshape_as(x),clips,zeros
        measurements={}
        per_head=[]
        def record(name,x,y,clips,zeros,reference_out,actual_out):
            row=measurements.setdefault(name,dict(error_sq=0.,reference_sq=0.,output_error_sq=0.,output_reference_sq=0.,clips=0,zeros=0,elements=0))
            row['error_sq']+=(y-x).square().sum().item()
            row['reference_sq']+=x.square().sum().item()
            row['output_error_sq']+=(actual_out-reference_out).square().sum().item()
            row['output_reference_sq']+=reference_out.square().sum().item()
            row['clips']+=clips;row['zeros']+=zeros;row['elements']+=x.numel()
        for si,ids in enumerate(evaluation):
            cos,sin=forward(ids)
            for li in range(nl):
                q,k,v=tensors(li,cos,sin,True)
                def attention(keys,values):
                    kh=keys.repeat_interleave(hq//hk,1).transpose(0,1)
                    vh=values.repeat_interleave(hq//hk,1).transpose(0,1)
                    logits=q.transpose(0,1)@kh.transpose(1,2)/(d**0.5)
                    mask=torch.ones(len(ids),len(ids),device='cuda',dtype=torch.bool).triu(1)
                    return logits.masked_fill(mask,-torch.inf).softmax(-1)@vh
                baseline=attention(k,v)
                for name,base in [('K',k),('V',v),('K_smoothed',k/lam[li])]:
                    a=base.abs().amax(0)
                    channel=a.amax(-1)/a.mean(-1).clamp_min(1e-20)
                    token=base.abs().amax(-1)
                    temporal=token.amax(0)/token.mean(0).clamp_min(1e-20)
                    for hi in range(hk):
                        per_head.append(dict(sample=si,layer=li,head=hi,tensor=name,channel_spread=channel[hi].item(),token_spread=temporal[hi].item()))
                for g in [128,64,32]:
                    restored,clips,zeros=dynamic(k,g)
                    record('K_dynamic_'+str(g),k,restored,clips,zeros,baseline,attention(restored,v))
                restored,clips,zeros=dynamic(k/lam[li],64)
                restored=restored*lam[li]
                record('K_smooth_dynamic_64',k,restored,clips,zeros,baseline,attention(restored,v))
                for name,inp,scale,factor in [('K_static_channel',k,sk_raw[li],1),('K_smooth_static_channel',k/lam[li],sk[li],lam[li])]:
                    restored,clips,zeros=quant(inp,scale)
                    restored=restored*factor
                    record(name,k,restored,clips,zeros,baseline,attention(restored,v))
                for g in [128,64,32]:
                    restored,clips,zeros=dynamic(v,g)
                    record('V_dynamic_'+str(g),v,restored,clips,zeros,baseline,attention(k,restored))
                restored,clips,zeros=quant(v,sv_static[li])
                record('V_static_channel',v,restored,clips,zeros,baseline,attention(k,restored))
            print('evaluation',si+1,len(evaluation),flush=True)
    for handle in handles:handle.remove()
    for row in measurements.values():
        row['relative_l2']=(row['error_sq']/row['reference_sq'])**.5
        row['attention_output_relative_l2']=(row['output_error_sq']/row['output_reference_sq'])**.5
        row['clipping_fraction']=row['clips']/row['elements']
        row['zero_fraction']=row['zeros']/row['elements']
    summaries={}
    for kind in ['K','V','K_smoothed']:
        rows=[r for r in per_head if r['tensor']==kind]
        summaries[kind]={field:sum(r[field] for r in rows)/len(rows) for field in ['channel_spread','token_spread']}
    fields=['relative_l2','attention_output_relative_l2','clipping_fraction','zero_fraction']
    result={'model':Path(args.model).name,'calibration_lengths':list(map(len,calibration)),
            'evaluation_lengths':list(map(len,evaluation)),
            'distribution':{name:{key:round(v,6) for key,v in row.items()} for name,row in summaries.items()},
            'measurements':{name:{key:round(row[key],6) for key in fields} for name,row in measurements.items()}}
    save(args.out/'results.json',result)
    table(args.out/'report.md','KV4 量化误差',['方案','相对 L2','Attention 输出相对 L2','截断比例','零值比例'],
        [[name,*[row[key] for key in fields]] for name,row in result['measurements'].items()],
        f'FP16 {Path(args.model).name}；校准 {len(calibration)} 段、共 {sum(map(len,calibration))} token，评估 {len(evaluation)} 段、共 {sum(map(len,evaluation))} token。每次只量化 K 或 V，其他量保持浮点，统计所有层和 KV head。')
    lines=['','| 张量 | 通道范围不均衡 C | token 峰值变化 R |','| --- | --- | --- |']
    for name,row in result['distribution'].items():
        lines.append(f"| {name} | {row['channel_spread']} | {row['token_spread']} |")
    lines+=['','先在每个样本、层和 KV head 上计算，再等权平均。C 为各通道最大幅度的最大值/均值，R 为各 token 最大幅度的最大值/均值。']
    with (args.out/'report.md').open('a') as f:f.write('\n'.join(lines)+'\n')
    publish(args.out,Path(__file__).parent)


if __name__=='__main__':main()
