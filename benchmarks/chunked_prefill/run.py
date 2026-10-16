"""One command: current prefill-first policy versus adapted mixed chunked policy."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from benchmarks.kv4.run import run_job


def report(out):
    lines=['# 混合 Chunked Prefill 小规模验收','',
        '先运行短请求；第 2 步后加入 512-token 长请求和另一条短请求。总共 32 个输出 token；batch≤4，token 预算 64，纯 decode 使用 graph。','',
        '| 权重 / KV | 策略 | 总耗时 ms | 输出 token/s | 原短请求 TPOT ms | 原短请求最大 token 间隔 ms | 长请求 TTFT ms | decode 推迟步数 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    results={}
    for mode in ['fp16-fp16','w4-kv4']:
        results[mode]={}
        for policy in ['prefill-first','mixed']:
            row=json.loads((out/f'{mode}-{policy}.json').read_text());results[mode][policy]=row
            deferred=row['runs'][0]['short_decode_deferred_steps']
            lines.append(f"| {mode} | {policy} | {row['total_ms']:.3f} | {row['output_tokens_per_s']:.2f} | "
                f"{row['short_tpot_ms']:.3f} | {row['short_max_token_gap_ms']:.3f} | {row['long_ttft_ms']:.3f} | {deferred} |")
    lines+=['','每个配置独立进程，预热一轮、测三轮取中位数；全部原始请求时间、输出、调度轨迹、graph replay 次数和显存记录见 JSON。',
        '验收检查实际存在混合步骤、64-token 预算、原短请求未被 prefill 推迟、纯 decode 使用 graph，以及完成/关闭后的释放。',
        'FP16/KV4 的混合 attention 另有数值回归；本工作负载测调度与延迟，不将 W4 的重复性波动视为调度输出必须逐位一致。',
        '这是三请求小规模测量，不能外推高并发吞吐或显存紧张时的抢占行为。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'summary.json').write_text(json.dumps({mode:{policy:{k:v for k,v in row.items() if k!='runs'}
        for policy,row in group.items()} for mode,group in results.items()},indent=2))


def main():
    p=argparse.ArgumentParser(description='混合 Chunked Prefill 三请求对照')
    p.add_argument('--gpu',default='1')
    p.add_argument('--samples',default='datasets/w4a16/samples.json')
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    out=args.out or Path('results/chunked_prefill')/datetime.now().strftime('%Y-%m-%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=True)
    (out/'run.json').write_text(json.dumps({**vars(args),'python':sys.executable,'seed':123,'repeats':3,
        'upstream':'dbf934fcb44cbc6b763517106153767d97719bf3',
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'dirty':bool(subprocess.check_output(['git','status','--porcelain'],text=True))},indent=2,default=str))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    for mode,model,kv in [('fp16-fp16','models/Qwen3-8B','fp16'),('w4-kv4','models/Qwen3-8B-W4-rtn','kv4')]:
        for policy in ['prefill-first','mixed']:
            name=f'{mode}-{policy}'
            cmd=[sys.executable,'-m','benchmarks.chunked_prefill.worker','--model',model,'--kv',kv,
                 '--samples',args.samples,'--out',str(out/f'{name}.json')]
            if policy=='mixed':cmd.append('--mixed')
            run_job(cmd,out/f'{name}.log',env)
    report(out)
    print(out/'report.md')


if __name__=='__main__':main()
