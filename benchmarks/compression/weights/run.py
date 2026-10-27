from benchmarks.common import publish
"""Compare four offline algorithms using full checkpoints and one fixed text set."""
import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from benchmarks.common import SAMPLES, run, worker, save, table

ALGORITHMS={'rtn':'rtn_w4a16','rtn-clip':'rtn_clip_w4a16',
            'awq':'awq_w4a16','awq-affine':'awq_affine_w4a16'}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gpu',default='1')
    p.add_argument('--model',default='models/Qwen3-8B')
    p.add_argument('--algorithms',nargs='+',choices=ALGORITHMS,default=list(ALGORITHMS))
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--calibration',default='datasets/w4a16/checkpoint_calibration.pt')
    p.add_argument('--samples',type=Path,default=SAMPLES)
    p.add_argument('--out',type=Path,default=Path('results/compression/weights'))
    args=p.parse_args();rows=[]
    with TemporaryDirectory() as temp:
        tasks=[('fp16',args.model,'fp16')]
        for algorithm in args.algorithms:
            checkpoint=Path(args.model+'-W4-'+algorithm)
            if args.prepare and not checkpoint.exists():
                print('准备',algorithm,flush=True)
                run([sys.executable,'-m','qslab.compression','--model',args.model,
                     '--algorithm',ALGORITHMS[algorithm],'--calibration',args.calibration,
                     '--out',str(checkpoint)],args.gpu)
            tasks.extend((algorithm,str(checkpoint),mode) for mode in ['fake','real'])
        for label,model,mode in tasks:
            print('测量',label,mode,flush=True)
            row=worker('benchmarks.compression.weights.evaluate',
                ['--model',model,'--mode',mode,'--samples',args.samples],args.gpu,Path(temp)/'row.json')
            rows.append({'algorithm':label,**row})
    save(args.out/'results.json',{'model':Path(args.model).name,'measurements':rows})
    samples=json.loads(args.samples.read_text())['samples']
    lengths=sorted({len(sample['token_ids']) for sample in samples})
    table(args.out/'report.md','离线压缩结果',['算法','分组','执行方式','PPL','权重 MiB','权重包 MiB'],
          [[r['algorithm'],r['group_size'] or '—',r['mode'],r['ppl'],r['weight_mib'],r['checkpoint_mib']] for r in rows],
          f'通过 HF 模型接口测量，不使用 KV 缓存。{len(samples)} 段输入，长度 {"/".join(map(str,lengths))} token，共 {rows[0]["prediction_tokens"]} 个预测目标。fake 在加载后恢复成浮点权重；real 执行打包 W4。')
    publish(args.out,Path(__file__).parent)


if __name__=='__main__':main()
