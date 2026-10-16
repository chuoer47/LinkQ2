"""One command runs independent references and the four engine combinations."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from benchmarks.kv4.run import run_job


MODES = ['fp16-fp16', 'fp16-kv4', 'w4-fp16', 'w4-kv4']


def report(output, modes):
    results = {mode: json.loads((output / f'{mode}.json').read_text()) for mode in modes}
    baseline = results.get('fp16-fp16')
    lines = ['# 单卡 eager 引擎验收', '',
             'Qwen3-8B；单请求；包含入队、调度、元数据准备、模型、KV 读写、采样与提交。', '',
             '| 权重 / KV | 输入 token | TTFT ms | TPOT ms | 总耗时 ms | 输出 token/s | TPOT 加速比 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for mode, row in results.items():
        for index, perf in enumerate(row['performance']):
            speed = baseline['performance'][index]['tpot_ms'] / perf['tpot_ms'] if baseline else None
            lines.append(f"| {mode} | {perf['context']} | {perf['ttft_ms']:.3f} | {perf['tpot_ms']:.3f} | "
                         f"{perf['e2e_ms']:.3f} | {perf['output_tokens_per_s']:.2f} | "
                         + (f'{speed:.3f}× |' if speed else '— |'))
    lines += ['', '| 权重 / KV | 权重 GiB | 固定 KV MiB | 常驻 allocated GiB | 启动峰值 GiB | close 后 MiB |',
              '|---|---:|---:|---:|---:|---:|']
    for mode, row in results.items():
        lines.append(f"| {mode} | {row['weight_bytes']/2**30:.3f} | {row['kv_bytes']/2**20:.3f} | "
                     f"{row['resident_allocated_bytes']/2**30:.3f} | {row['startup_peak_allocated_bytes']/2**30:.3f} | "
                     f"{row['allocated_bytes_after_close']/2**20:.3f} |")
    lines += ['', '| 权重 / KV | 参考 | 配对位置数 | teacher-forced top1 | logits 相对 L2 | 自由生成 token 一致率 | 生命周期验收 |',
              '|---|---|---:|---:|---:|---:|---|']
    for mode, row in results.items():
        q = row['quality']
        lines.append(f"| {mode} | HF {'W4 fake' if mode.startswith('w4') else 'FP16'} | {q['positions']} | "
                     f"{q['teacher_forced_top1_agreement']*100:.2f}% | {q['relative_l2']:.6f} | "
                     f"{q['free_generation_token_agreement']*100:.2f}% | {'通过' if all(row['lifecycle'].values()) else '失败'} |")
    lines += ['', 'W4 与相同 checkpoint 的 HF fake 配对，KV4 与对应浮点 KV 参考配对。这里验证执行正确性，不替代量化 PPL 实验。',
              '每个形状完整预热一次、测量三次取中位数；实际次数见 run.json。计时关闭前缀复用，排除加载、tokenize/detokenize、参考对照和首次编译。',
              'TPOT 为首 token 到最后 token 的平均间隔；吞吐为输出 token 数除以整次请求耗时。',
              '四种配置固定 4×256 token 的 KV 容量，显存 allocated 不含 CUDA 上下文；启动峰值含加载和引擎预热。close 后与预先初始化的 cuBLAS 工作区基线比较。',
              '本轮不验收并发调度、CUDA graph、投机推理或 HTTP 吞吐。']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (output / 'summary.json').write_text(json.dumps({mode: {
        'passed': row['passed'], 'quality': row['quality'], 'lifecycle': row['lifecycle'],
        'weight_bytes': row['weight_bytes'], 'kv_bytes': row['kv_bytes'],
        'performance': [{key: value for key, value in perf.items() if key != 'runs'}
                        for perf in row['performance']]} for mode, row in results.items()}, indent=2))


def main():
    p = argparse.ArgumentParser(description='单卡推理引擎小规模验收与性能对照')
    p.add_argument('--gpu', default='1')
    p.add_argument('--model', default='models/Qwen3-8B')
    p.add_argument('--w4-model', default='models/Qwen3-8B-W4-rtn')
    p.add_argument('--calibration', default='datasets/kv4/qwen3-8b.pt')
    p.add_argument('--samples', type=Path, default=Path('datasets/w4a16/samples.json'))
    p.add_argument('--modes', nargs='+', choices=MODES, default=MODES)
    p.add_argument('--tokens', type=int, default=16)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--contexts', nargs='+', type=int, default=[128, 512])
    p.add_argument('--max-model-len', type=int, default=768)
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    if (args.tokens < 2 or args.repeats < 1 or min(args.contexts) < 1
            or max(args.contexts) + args.tokens > args.max_model_len
            or args.max_model_len > 1024):
        p.error('需要 tokens>=2、repeats>=1，输入+输出不超过 max-model-len<=1024')
    output = args.out or Path('results/engine') / datetime.now().strftime('%Y-%m-%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=True)
    ids = json.loads(args.samples.read_text())['samples'][0]['token_ids']
    if len(ids) < max(args.contexts):
        p.error('samples 中的首段 token 不足')
    (output / 'inputs.json').write_text(json.dumps({'token_ids': ids[:max(args.contexts)]}))
    metadata = {**vars(args), 'python': sys.executable, 'seed': 123,
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True))}
    (output / 'run.json').write_text(json.dumps(metadata, indent=2, default=str))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    references = {}
    for weights in dict.fromkeys(mode.split('-')[0] for mode in args.modes):
        target = output / f'reference-{weights}.pt'
        cmd = [sys.executable, '-m', 'benchmarks.engine.reference', '--model',
               args.w4_model if weights == 'w4' else args.model,
               '--tokens', str(args.tokens), '--out', str(target)]
        if weights == 'w4':
            cmd.append('--quantized')
        run_job(cmd, output / f'reference-{weights}.log', env)
        references[weights] = target
    for mode in args.modes:
        weights, kv = mode.split('-')
        run_job([sys.executable, '-m', 'benchmarks.engine.worker',
            '--model', args.w4_model if weights == 'w4' else args.model,
            '--kv', kv, '--calibration', args.calibration,
            '--reference', str(references[weights]), '--inputs', str(output / 'inputs.json'),
            '--tokens', str(args.tokens), '--repeats', str(args.repeats),
            '--contexts', *map(str, args.contexts), '--max-model-len', str(args.max_model_len),
            '--out', str(output / f'{mode}.json')], output / f'{mode}.log', env)
    report(output, args.modes)
    print(output / 'report.md')


if __name__ == '__main__':
    main()
