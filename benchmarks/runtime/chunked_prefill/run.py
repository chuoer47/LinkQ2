from benchmarks.common import publish
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from benchmarks.common import worker, save, table

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',default='1')
    p.add_argument('--out',type=Path,default=Path('results/runtime/chunked_prefill'))
    args=p.parse_args();rows=[]
    with TemporaryDirectory() as temp:
        for quantized in [False,True]:
            for mixed in [False,True]:
                flags=(['--quantized'] if quantized else [])+(['--mixed'] if mixed else [])
                rows.append(worker('benchmarks.runtime.chunked_prefill.worker',flags,args.gpu,Path(temp)/'row.json'))
    save(args.out/'results.json',{'measurements':rows})
    table(args.out/'report.md','分块预填充调度',['配置','调度','整批 ms','输出 token/s'],
          [[r['configuration'],'混合' if r['mixed'] else 'prefill 优先',r['e2e_ms'],r['output_tokens_per_s']] for r in rows],
          '一个短请求执行两步后，再加入一个长请求和一个短请求；JSON 保留各请求的首 token 延迟和最大 token 间隔。')
    lines=['','| 配置 | 调度 | 请求 | TTFT ms | TPOT ms/token | 最大 token 间隔 ms |',
           '| --- | --- | --- | --- | --- | --- |']
    for row in rows:
        for i,name in enumerate(['短请求 1','长请求','短请求 2']):
            values=[row['requests'][i][key]
                    for key in ['ttft_ms','tpot_ms','max_token_gap_ms']]
            lines.append('| '+' | '.join(map(str,[row['configuration'],'混合' if row['mixed'] else 'prefill 优先',name,*values]))+' |')
    with (args.out/'report.md').open('a') as f:f.write('\n'.join(lines)+'\n')
    publish(args.out,Path(__file__).parent)


if __name__=='__main__':main()
