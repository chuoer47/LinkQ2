"""Run the small benchmark sequentially in independent worker processes."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys


def report(output):
    rows = {p.stem: json.loads(p.read_text()) for p in output.glob('*.json')
            if p.stem == 'fp16' or p.stem.endswith(('_fake', '_real'))}
    rows = dict(sorted(rows.items(), key=lambda item: (item[0] != 'fp16', item[0])))
    baseline = rows['fp16']
    lines = ['# W4A16 实测结果', '',
             'HF eager，FP16，batch=1；PPL 固定 8×512 token（4088 个预测目标）。', '',
             '| 模式 | 权重 GiB | PPL | prefill 128 ms | prefill 512 ms | decode ms/token | decode 加速比 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, row in rows.items():
        pre = row['prefill']
        latency = row['decode']['latency_median_ms']
        lines.append(f"| {name} | {row['weight_storage_bytes']/2**30:.3f} | "
                     f"{row['quality']['ppl']:.6f} | {pre[0]['latency_median_ms']:.3f} | "
                     f"{pre[1]['latency_median_ms']:.3f} | {latency:.3f} | "
                     f"{baseline['decode']['latency_median_ms']/latency:.3f}× |")
    lines += ['', '加速比 = FP16 延迟 / 当前延迟；大于 1 才表示加速。权重存储与 PyTorch allocated/reserved、峰值分别记录于 JSON。',
              '延迟为同步主机墙钟时间，包含 HF 调度与 lm_head；decode 使用普通 HF KV cache，固定输入 token。', '']
    for name, row in rows.items():
        if row['mode'] == 'real':
            fake = rows[name.replace('_real', '_fake')]
            lines.append(f"- {name}：替换并执行 {row['executed_layers']} 层；real−fake PPL = "
                         f"{row['quality']['ppl']-fake['quality']['ppl']:+.6f}；probe 相对 L2 = "
                         f"{row['vs_fake_probe']['relative_l2']:.6f}；释放模型后 allocated = "
                         f"{row['released_memory']['allocated_bytes']/2**20:.2f} MiB。")
    lines += ['', f"FP16 基线释放后仍有 {baseline['released_memory']['allocated_bytes']/2**20:.2f} MiB allocated（PyTorch/cuBLAS 工作区）；模型释放需与同环境的预热基线比较。"]
    port = output / 'port_comparison.json'
    if port.exists():
        cases = json.loads(port.read_text())['cases']
        lines += ['', '## 与原 GemLite 比较', '',
                  f"{len(cases)} 组 packed/scales/zeros 完全一致；输出最大相对 L2 = "
                  f"{max(row['relative_l2'] for row in cases):.6f}。", '',
                  '| N×K | M | 移植 GPU μs | 原包 GPU μs |', '|---|---:|---:|---:|']
        for row in cases:
            if row['m'] == 1:
                lines.append(f"| {row['n']}×{row['k']} | 1 | {row['ported_gpu_ms']*1000:.2f} | "
                             f"{row['upstream_gpu_ms']*1000:.2f} |")
    operators = output / 'operators.json'
    if operators.exists():
        lines += ['', '## M=1 单算子', '',
                  '| N×K | FP16 GPU μs | W4 GPU μs | GPU 加速比 | FP16 主机 μs | W4 主机 μs |',
                  '|---|---:|---:|---:|---:|---:|']
        for row in json.loads(operators.read_text())['cases']:
            if row['m'] == 1:
                lines.append(f"| {row['n']}×{row['k']} | {row['fp16_gpu_ms']*1000:.2f} | "
                             f"{row['real_gpu_ms']*1000:.2f} | {row['gpu_speedup_vs_fp16']:.2f}× | "
                             f"{row['fp16_host_ms']*1000:.2f} | {row['real_host_ms']*1000:.2f} |")
        lines += ['', 'GPU 数据为 CUDA graph 热权重重放；主机数据包含调用与同步开销。单算子优势不能直接换算为 HF decode 收益。']
    (output / '报告.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='小规模 W4A16 性能、显存和 real/fake 精度对照')
    parser.add_argument('--model', default='models/Qwen3-8B')
    parser.add_argument('--artifacts', type=Path, default=Path('artifacts'))
    parser.add_argument('--algorithms', nargs='+', default=['rtn', 'awq'])
    parser.add_argument('--samples', default='datasets/w4a16/samples.json')
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    output = args.out or Path('results/w4a16') / datetime.now().strftime('%Y-%m-%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.samples, output / 'samples.json')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, TOKENIZERS_PARALLELISM='false')
    (output / 'run.json').write_text(json.dumps({**vars(args), 'python': sys.executable,
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True))},
        default=str, indent=2))
    jobs = [('fp16', None, 'fp16')]
    jobs += [(f'{algorithm}_{mode}', str(args.artifacts / algorithm), mode)
             for algorithm in args.algorithms for mode in ('fake', 'real')]
    for name, artifact, mode in jobs:
        cmd = [sys.executable, '-m', 'benchmarks.w4a16.worker', '--model', args.model,
               '--samples', args.samples, '--mode', mode, '--out', str(output / f'{name}.json')]
        if artifact:
            cmd += ['--artifact', artifact]
        print(f'{name}: {output / (name + ".log")}', flush=True)
        with (output / f'{name}.log').open('w') as log:
            child = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = child.wait()
            except KeyboardInterrupt:
                child.terminate()
                child.wait()
                raise
            if code:
                raise RuntimeError(f'{name} failed; see {output / (name + ".log")}')
    report(output)
    print(output / '报告.md')


if __name__ == '__main__':
    main()
