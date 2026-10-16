"""Run official tests, shape checks, and paired full-model PPL in isolated processes."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def run_job(command, log, env):
    print(log.name, flush=True)
    with log.open('w') as file:
        child = subprocess.Popen(command, env=env, stdout=file, stderr=subprocess.STDOUT)
        try:
            code = child.wait()
        except KeyboardInterrupt:
            child.terminate()
            child.wait()
            raise
    if code:
        raise RuntimeError(f'check failed; stopped: {log}')


def report(output, checkpoints):
    rows = []
    for checkpoint in checkpoints:
        name = Path(checkpoint).name
        fake = json.loads((output / (name + '_fake.json')).read_text())
        real = json.loads((output / (name + '_real.json')).read_text())
        assert real['executed_layers'] == real['replaced_layers'], name
        gap = real['quality']['ppl'] / fake['quality']['ppl'] - 1
        assert abs(gap) < 0.01, f'{name}: real/fake PPL differs by more than 1%'
        rows.append(dict(checkpoint=checkpoint, fake_ppl=fake['quality']['ppl'],
                         marlin_ppl=real['quality']['ppl'], ppl_delta=real['quality']['ppl']-fake['quality']['ppl'],
                         relative_ppl_gap=gap, layers=real['executed_layers'],
                         weight_bytes=real['weight_storage_bytes'], probe=real['vs_fake_probe'],
                         decode=real['vs_fake_decode'], released_memory=real['released_memory']))
    baseline = json.loads((output / 'fp16.json').read_text())
    samples = json.loads((output / 'samples.json').read_text())['samples']
    summary = {'fp16_ppl': baseline['quality']['ppl'], 'samples': len(samples),
               'prediction_targets': sum(len(s['token_ids']) - 1 for s in samples), 'checkpoints': rows}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    cases = json.loads((output / 'operators.json').read_text())['cases']
    lines = ['# Marlin 官方接入验证', '',
             f"整模 PPL：{len(samples)} 段，{summary['prediction_targets']} 个预测目标；FP16 基线 PPL：{summary['fp16_ppl']:.6f}。", '',
             '| checkpoint | 量化层 | fake PPL | Marlin PPL | 差值 |', '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {Path(row['checkpoint']).name} | {row['layers']} | {row['fake_ppl']:.6f} | {row['marlin_ppl']:.6f} | {row['ppl_delta']:+.6f} |")
    lines += ['', f"单算子 {len(cases)} 组：官方 packed/scales 全部一致；最大平均绝对相对误差 {max(r['mean_abs_relative'] for r in cases):.6f}。", '',
              '| N×K | M | FP16 GPU μs | Marlin GPU μs | 加速比 |', '|---|---:|---:|---:|---:|']
    for row in cases:
        lines.append(f"| {row['n']}×{row['k']} | {row['m']} | {row['fp16_gpu_ms']*1000:.2f} | {row['marlin_gpu_ms']*1000:.2f} | {row['gpu_speedup_vs_fp16']:.2f}× |")
    lines += ['', '延迟为 CUDA graph 热权重单算子测量；不代表整模 decode 加速。本轮整模仅验证 PPL、短 decode、覆盖率和显存释放。']
    (output / '报告.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='官方 Marlin 数值与整模 PPL 验证')
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--model', default='models/Qwen3-8B')
    parser.add_argument('--checkpoints', nargs='+', default=['models/Qwen3-8B-W4-rtn', 'models/Qwen3-8B-W4-awq'])
    parser.add_argument('--samples', type=Path, default=Path('datasets/w4a16/samples.json'))
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    output = args.out or Path('results/marlin') / datetime.now().strftime('%Y-%m-%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.samples, output / 'samples.json')
    (output / 'run.json').write_text(json.dumps({**vars(args), 'python': sys.executable,
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True))}, default=str, indent=2))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    run_job([sys.executable, 'third_party/marlin/test.py'], output / 'official_tests.log', env)
    run_job([sys.executable, '-m', 'benchmarks.marlin.operator', '--out', str(output / 'operators.json')],
            output / 'operators.log', env)
    jobs = [('fp16', args.model, 'fp16')]
    jobs += [(f'{Path(path).name}_{mode}', path, mode) for path in args.checkpoints for mode in ('fake', 'real')]
    for name, path, mode in jobs:
        run_job([sys.executable, '-m', 'benchmarks.w4a16.worker', '--backend', 'marlin',
                 '--model', path, '--samples', str(args.samples), '--mode', mode, '--quality-only',
                 '--out', str(output / (name + '.json'))], output / (name + '.log'), env)
    report(output, args.checkpoints)
    print(output / '报告.md')


if __name__ == '__main__':
    main()
