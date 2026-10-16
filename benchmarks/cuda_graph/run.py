"""Small paired benchmark; reuse the established engine measurement worker."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from benchmarks.kv4.run import run_job


MODES = ['fp16-fp16', 'w4-kv4']


def report(out):
    import torch
    rows = {f'{mode}-{execution}': json.loads((out / f'{mode}-{execution}.json').read_text())
            for mode in MODES for execution in ['eager', 'graph']}
    pairs = {}
    for mode in MODES:
        eager = torch.load(out / f'{mode}-eager.logits.pt', weights_only=True).float()
        graph = torch.load(out / f'{mode}-graph.logits.pt', weights_only=True).float()
        pairs[mode] = {'relative_l2': float((eager-graph).norm()/eager.norm()),
            'top1_agreement': float((eager.argmax(-1) == graph.argmax(-1)).float().mean()),
            'max_abs_error': float((eager-graph).abs().max())}
        pairs[mode]['passed'] = pairs[mode]['relative_l2'] <= 0.001 and pairs[mode]['top1_agreement'] >= 0.99
    lines = ['# CUDA Graph 配对对照', '',
        'Qwen3-8B；相同输入、输出预算、KV 容量和采样；单请求；decode 捕获，prefill/最后 lm_head/采样在图外。', '',
        '| 权重 / KV | 输入 | eager TTFT ms | graph TTFT ms | eager TPOT ms | graph TPOT ms | graph 加速比 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for mode in MODES:
        for e, g in zip(rows[mode+'-eager']['performance'], rows[mode+'-graph']['performance']):
            lines.append(f"| {mode} | {e['context']} | {e['ttft_ms']:.3f} | {g['ttft_ms']:.3f} | "
                         f"{e['tpot_ms']:.3f} | {g['tpot_ms']:.3f} | {e['tpot_ms']/g['tpot_ms']:.3f}× |")
    lines += ['', '| 输入 | FP16+KV16 graph TPOT ms | W4+KV4 graph TPOT ms | W4+KV4 相对加速比 |',
              '|---:|---:|---:|---:|']
    for b, q in zip(rows['fp16-fp16-graph']['performance'], rows['w4-kv4-graph']['performance']):
        lines.append(f"| {b['context']} | {b['tpot_ms']:.3f} | {q['tpot_ms']:.3f} | {b['tpot_ms']/q['tpot_ms']:.3f}× |")
    lines += ['', '| 配置 | 常驻 allocated GiB | 运行峰值 GiB | KV MiB | graph replay 次数 | 关闭后 MiB |',
              '|---|---:|---:|---:|---:|---:|']
    for key, r in rows.items():
        lines.append(f"| {key} | {r['resident_allocated_bytes']/2**30:.3f} | {r['runtime_peak_allocated_bytes']/2**30:.3f} | "
                     f"{r['kv_bytes']/2**20:.3f} | {r['graph_replays']} | {r['allocated_bytes_after_close']/2**20:.3f} |")
    lines += ['', '| 权重 / KV | graph/eager logits 相对 L2 | top1 一致率 |', '|---|---:|---:|']
    for mode, pair in pairs.items():
        lines.append(f"| {mode} | {pair['relative_l2']:.8f} | {pair['top1_agreement']*100:.2f}% |")
    lines += ['', '各 3 个短提示×16 个参考位置；固定相同历史比较 graph/eager，而非比较两种量化精度。',
              '每个输入形状预热一次、测量三次取中位数。计时包含调度、元数据、模型、KV、lm_head、采样和提交，排除加载与 tokenize/detokenize。',
              '完整 TTFT、TPOT、总延迟、输出吞吐、权重字节数、启动峰值及原始三次结果见各 JSON。',
              'cuBLAS 按流保留库工作区；关闭后的分配与各进程预初始化工作区基线比较，模型、KV 和 graph 缓冲区需释放。']
    (out / 'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    summary = {'pairs': pairs, 'passed': all(r['passed'] for r in rows.values()) and all(p['passed'] for p in pairs.values())}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    if not summary['passed']:
        raise RuntimeError('graph/eager acceptance failed; inspect results')


def main():
    p = argparse.ArgumentParser(description='FP16+KV16 与 W4+KV4 的 eager/CUDA Graph 对照')
    p.add_argument('--gpu', default='1')
    p.add_argument('--model', default='models/Qwen3-8B')
    p.add_argument('--w4-model', default='models/Qwen3-8B-W4-rtn')
    p.add_argument('--calibration', default='datasets/kv4/qwen3-8b.pt')
    p.add_argument('--references', type=Path, help='复用已有 reference-fp16.pt/reference-w4.pt')
    p.add_argument('--samples', type=Path, default=Path('datasets/w4a16/samples.json'))
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    out = args.out or Path('results/cuda_graph') / datetime.now().strftime('%Y-%m-%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    metadata = {**vars(args), 'python': sys.executable, 'tokens': 16, 'repeats': 3,
                'contexts': [128,512], 'kv_slots':1024, 'seed':123,
                'git_head': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'dirty': bool(subprocess.check_output(['git','status','--porcelain'],text=True))}
    (out/'run.json').write_text(json.dumps(metadata,indent=2,default=str))
    ids = json.loads(args.samples.read_text())['samples'][0]['token_ids'][:512]
    (out/'inputs.json').write_text(json.dumps({'token_ids':ids}))
    env = dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    for weights, model in [('fp16', args.model),('w4', args.w4_model)]:
        target = out/f'reference-{weights}.pt'
        if args.references:
            shutil.copyfile(args.references/target.name,target)
        else:
            cmd = [sys.executable,'-m','benchmarks.engine.reference','--model',model,'--out',str(target)]
            if weights=='w4': cmd.append('--quantized')
            run_job(cmd,out/f'reference-{weights}.log',env)
    for mode in MODES:
        weights,kv = mode.split('-')
        for execution in ['eager','graph']:
            name=f'{mode}-{execution}'
            run_job([sys.executable,'-m','benchmarks.engine.worker','--model',args.w4_model if weights=='w4' else args.model,
                '--kv',kv,'--execution',execution,'--probe-logits','--calibration',args.calibration,
                '--reference',str(out/f'reference-{weights}.pt'),'--inputs',str(out/'inputs.json'),
                '--out',str(out/f'{name}.json')],out/f'{name}.log',env)
    report(out)
    print(out/'report.md')


if __name__=='__main__':
    main()
