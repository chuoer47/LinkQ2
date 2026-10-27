"""Nsight capture using its command-line tools, with concise CSV/JSON summaries."""
import argparse, csv, json, os, shutil, subprocess, sys
from pathlib import Path
from benchmarks.common import environment, save, table, publish

def tool(name, supplied):
    found=supplied or shutil.which(name) or str(Path.home()/'.local/bin'/name)
    if not Path(found).is_file():raise FileNotFoundError('需要可执行的 '+name+'，可用 --'+name+' 指定路径')
    return found

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gpu',default='1')
    p.add_argument('--target',choices=['layer','engine'],default='layer')
    p.add_argument('--tool',choices=['systems','compute'],default='systems')
    p.add_argument('--nsys');p.add_argument('--ncu')
    p.add_argument('--context',type=int,default=512)
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    args.out=args.out or Path('results/runtime/profile')/args.target
    if args.tool=='compute':args.out=args.out/'compute'
    args.out.mkdir(parents=True,exist_ok=True);rows=[]
    for kv in ['fp16','kv4']:
        prefix=args.out/(args.target+'-'+kv)
        workload=([sys.executable,'-m','benchmarks.runtime.profile.layer','--mode',kv,
                   '--lengths',str(args.context),'--steps','3'] if args.target=='layer' else
                  [sys.executable,'-m','benchmarks.runtime.profile.engine','--kv',kv,'--context',str(args.context)])
        if args.tool=='systems':
            binary=tool('nsys',args.nsys)
            command=[binary,'profile','--trace','cuda,nvtx','--sample','none','--cpuctxsw','none',
                '--capture-range','cudaProfilerApi','--capture-range-end','stop','--force-overwrite','true',
                '--output',str(prefix),*workload]
        else:
            binary=tool('ncu',args.ncu)
            workload+=['--execution','eager','--case','decode'] if args.target=='layer' else []
            command=[binary,'--profile-from-start','off','--clock-control','none','--cache-control','none',
                '--set','basic','--launch-count','1','--export',str(prefix),'--force-overwrite',*workload]
        print('采集',args.target,kv,args.tool,flush=True)
        completed=subprocess.run(command,env=environment(args.gpu),capture_output=True,text=True)
        if completed.returncode:raise RuntimeError(completed.stdout[-3000:]+completed.stderr[-3000:])
        if args.tool=='systems':
            result=subprocess.run([binary,'stats','--force-export','true','--force-overwrite','true','--report','cuda_gpu_kern_sum:base','--format','csv',
                '--output',str(prefix),str(prefix)+'.nsys-rep'],capture_output=True,text=True)
            if result.returncode:raise RuntimeError(result.stdout+result.stderr)
            csv_path=Path(str(prefix)+'_cuda_gpu_kern_sum_base.csv')
            with csv_path.open() as f:
                for row in list(csv.DictReader(f))[:10]:
                    rows.append({'kv':kv,'kernel':row['Name'],
                        'calls':int(row['Instances']),
                        'total_ms':round(float(row['Total Time (ns)'])/1e6,3)})
            csv_path.unlink()
            sqlite=Path(str(prefix)+'.sqlite')
            if sqlite.exists():sqlite.unlink()
    if args.tool=='systems':
        save(args.out/'results.json',{'target':args.target,'context_tokens':args.context,'kernels':rows})
        table(args.out/'report.md','Nsight 内核耗时摘要',['KV','内核','调用次数','总 GPU 时间 ms'],
            [[r['kv'],r['kernel'],r['calls'],r['total_ms']] for r in rows],
            '同一内核的模板配置合并统计，每种 KV 列出总耗时前十项。target='+args.target+'，上下文 '+str(args.context)+' token。各内核的总时间包含采集区间内多次调用，不能当作单次延迟；按 NVTX 区间查看细节请打开同目录的 .nsys-rep。')

        publish(args.out,Path(__file__).parent,'LAYER.md' if args.target=='layer' else 'ENGINE.md')


if __name__=='__main__':main()
