from benchmarks.common import publish
"""Full-model KV4 quality at a fixed cache capacity."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from benchmarks.common import worker, save, table

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gpu',default='1')
    p.add_argument('--out',type=Path,default=Path('results/runtime/kv4'))
    args=p.parse_args();rows=[]
    with TemporaryDirectory() as temp:
        for kv in ['fp16','kv4']:
            print('测量',kv,flush=True)
            rows.append(worker('benchmarks.runtime.kv4.worker',['--mode',kv],args.gpu,Path(temp)/'row.json'))
    save(args.out/'results.json',{'model':'Qwen3-8B','weight_dtype':'fp16','kv_capacity_tokens':1024,'measurements':rows})
    table(args.out/'report.md','KV4 整模质量',['KV','PPL','KV MiB'],[[r['kv'],r['ppl'],r['kv_mib']] for r in rows],
          '固定 FP16 权重和相同校准参数。8×512 token；首 token 预填充，后续按 32 行因果验证读取缓存，量化误差会逐层传播。')
    publish(args.out,Path(__file__).parent)


if __name__=='__main__':main()
