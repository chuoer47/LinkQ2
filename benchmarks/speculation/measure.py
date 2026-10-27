"""Shared small comparison, with no extra model or request abstraction."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from benchmarks.common import worker, save, table

def parser(name):
    p=argparse.ArgumentParser()
    p.add_argument('--gpu',default='1')
    p.add_argument('--max-tokens',type=int,default=256)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--limit',type=int,default=2)
    p.add_argument('--out',type=Path,default=Path('results/speculation')/name)
    return p

def compare(args,tasks):
    rows=[]
    with TemporaryDirectory() as temp:
        for name,flags in tasks:
            print('测量',name,flush=True)
            row=worker('benchmarks.speculation.worker',[*flags,'--max-tokens',args.max_tokens,
                '--repeats',args.repeats,'--limit',args.limit],args.gpu,Path(temp)/'row.json')
            rows.append({'configuration':name,**row})
    baseline=None
    for row in rows:
        if row['method']=='baseline':baseline={m['case']:m for m in row['measurements']}
        for m in row['measurements']:
            m['tpot_speedup']=round(baseline[m['case']]['tpot_ms']/m['tpot_ms'],3)
    save(args.out/'results.json',{'max_output_tokens':args.max_tokens,'measurements':rows})
    table(args.out/'report.md','投机推理性能',['配置','题目','实际输出 token','总耗时 ms','TPOT ms/token','每次验证提交 token','接受率','TPOT 加速比'],
        [[r['configuration'],m['case'],m['output_tokens'],m['total_ms'],m['tpot_ms'],m['mean_len'] if r['method']!='baseline' else '—',m.get('acceptance_rate','—') if r['method']!='baseline' else '—',m['tpot_speedup']]
         for r in rows for m in r['measurements']],
        '所有输入使用相同非 thinking 模板，greedy，遇 EOS 停止。输出长度不同时，同时看实际 token 数和 TPOT；这里不计算任务正确率。')
    return rows
