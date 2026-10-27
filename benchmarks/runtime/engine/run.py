from benchmarks.common import publish
"""Four weight/cache combinations using the same request sizes."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from benchmarks.common import worker, save, table

def collect(tasks,args):
    rows=[]
    with TemporaryDirectory() as temp:
        for name,model,kv,graph in tasks:
            print('测量',name,flush=True)
            result=worker('benchmarks.runtime.engine.worker',['--model',model,'--kv',kv,
                '--graph' if graph else '--no-graph','--contexts',*args.contexts,
                '--tokens',args.tokens,'--repeats',args.repeats],args.gpu,Path(temp)/'row.json')
            rows.append({'configuration':name,**result})
    return rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gpu',default='1')
    p.add_argument('--contexts',nargs='+',type=int,default=[128,512])
    p.add_argument('--tokens',type=int,default=32)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--out',type=Path,default=Path('results/runtime/engine'))
    args=p.parse_args()
    rows=collect([(f'{weight}-{kv}',model,kv,True) for weight,model in
        [('fp16','models/Qwen3-8B'),('w4','models/Qwen3-8B-W4-rtn')] for kv in ['fp16','kv4']],args)
    save(args.out/'results.json',{'measurements':rows})
    table(args.out/'report.md','推理引擎性能',['配置','输入 token','TTFT ms','TPOT ms/token','总耗时 ms','KV MiB'],
          [[r['configuration'],m['input_tokens'],m['ttft_ms'],m['tpot_ms'],m['e2e_ms'],r['kv_mib']]
           for r in rows for m in r['measurements']],
          '开启 decode Graph 和 64 token 分块 prefill。计时包含调度、模型、KV、采样与提交，不含模型加载和分词。')
    publish(args.out,Path(__file__).parent)


if __name__=='__main__':main()
