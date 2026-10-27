from benchmarks.common import publish
"""Compare eager and Graph without turning numerical observations into performance gates."""
import argparse
from pathlib import Path
from benchmarks.common import save, table
from benchmarks.runtime.engine.run import collect

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gpu',default='1')
    p.add_argument('--contexts',nargs='+',type=int,default=[128,512])
    p.add_argument('--tokens',type=int,default=32)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--out',type=Path,default=Path('results/runtime/cuda_graph'))
    args=p.parse_args()
    rows=collect([(f'{name}-'+('graph' if graph else 'eager'),model,kv,graph)
        for name,model,kv in [('fp16-fp16','models/Qwen3-8B','fp16'),('w4-kv4','models/Qwen3-8B-W4-rtn','kv4')]
        for graph in [False,True]],args)
    save(args.out/'results.json',{'measurements':rows})
    table(args.out/'report.md','CUDA Graph 性能',['配置','输入 token','TTFT ms','TPOT ms/token','总耗时 ms'],
          [[r['configuration'],m['input_tokens'],m['ttft_ms'],m['tpot_ms'],m['e2e_ms']]
           for r in rows for m in r['measurements']],
          '两种执行方式使用相同输入与输出预算。Graph 只捕获模型和 KV 操作，prefill、lm_head 与采样在图外。')
    publish(args.out,Path(__file__).parent)


if __name__=='__main__':main()
