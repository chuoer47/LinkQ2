"""One command: small independent calibration and paired runtime KV measurements."""
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
        raise RuntimeError(f'failed; inspect {log}')


def report(output):
    baseline = json.loads((output / 'fp16.json').read_text())
    quantized = json.loads((output / 'kv4.json').read_text())
    summary = {'fp16_ppl': baseline['quality']['ppl'], 'kv4_ppl': quantized['quality']['ppl'],
               'ppl_delta': quantized['quality']['ppl'] - baseline['quality']['ppl'],
               'relative_ppl_change': quantized['quality']['ppl']/baseline['quality']['ppl']-1,
               'kv_memory_compression': baseline['kv_bytes']/quantized['kv_bytes'],
               'prediction_targets': quantized['quality']['tokens'],
               'executed_attention_layers': quantized['executed_attention_layers']}
    lines = ['# KV4 小规模整模对照', '', 'Qwen3-8B，固定 FP16 权重、相同 SmoothAttention；单请求，普通 eager。', '',
             '| KV | PPL | 固定 1024 slot 的 KV MiB |', '|---|---:|---:|']
    for name, row in [('FP16', baseline), ('INT4', quantized)]:
        lines.append(f"| {name} | {row['quality']['ppl']:.6f} | {row['kv_bytes']/2**20:.3f} |")
    lines += ['', f"预测目标 {summary['prediction_targets']}；PPL 变化 {summary['relative_ppl_change']*100:+.3f}%；KV 存储压缩 {summary['kv_memory_compression']:.3f}×。", '',
              '| 上下文 | FP16 prefill ms | KV4 prefill ms | FP16 decode ms/token | KV4 decode ms/token | decode 加速比 |',
              '|---:|---:|---:|---:|---:|---:|']
    for index, context in enumerate([128, 512]):
        b, q = baseline['decode'][index], quantized['decode'][index]
        speedup = b['median_ms_per_token']/q['median_ms_per_token']
        lines.append(f"| {context} | {baseline['prefill'][index]['median_ms']:.3f} | {quantized['prefill'][index]['median_ms']:.3f} | {b['median_ms_per_token']:.3f} | {q['median_ms_per_token']:.3f} | {speedup:.3f}× |")
    lines += ['', '速度为同步主机墙钟，包含模型、KV 写入/读取和最后位置 lm_head；不含调度、采样和上下文元数据准备。',
              'PPL 首个 token prefill，后续按 32 query 的 causal verify 路径读取 KV；decode 延迟单独使用逐 token 路径。',
              '两种 KV 均使用 256 token 分页及相同容量；FP16 对照调用官方 FlashAttention paged KV API，KV4 调用项目 Triton 算子。',
              'prefill 为各自实际 attention 实现：FP16 使用官方 cache API，KV4 写入后使用 FlashAttention varlen。',
              '该实验不包含 W4 权重量化、CUDA graph、投机接受率或多请求吞吐。']
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    (output / '报告.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='KV4 精度、速度与显存对照')
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--model', default='models/Qwen3-8B')
    parser.add_argument('--calibration', type=Path, default=Path('datasets/kv4/qwen3-8b.pt'))
    parser.add_argument('--calibration-tokens', default='datasets/w4a16/checkpoint_calibration.pt')
    parser.add_argument('--samples', default='datasets/w4a16/samples.json')
    parser.add_argument('--v-group', type=int, default=64)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    output = args.out or Path('results/kv4') / datetime.now().strftime('%Y-%m-%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.samples, output / 'samples.json')
    (output / 'run.json').write_text(json.dumps({**vars(args), 'python': sys.executable,
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True))}, default=str, indent=2))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    if not args.calibration.exists():
        run_job([sys.executable, 'scripts/build_smooth_kv.py', '--model', args.model,
                 '--calib', args.calibration_tokens, '--n-seqs', '8', '--seq-len', '512',
                 '--out', str(args.calibration)], output / 'calibration.log', env)
    for mode in ['fp16', 'kv4']:
        run_job([sys.executable, '-m', 'benchmarks.kv4.worker', '--model', args.model,
                 '--calibration', str(args.calibration), '--samples', args.samples,
                 '--mode', mode, '--v-group', str(args.v_group), '--out', str(output / (mode + '.json'))],
                output / (mode + '.log'), env)
    report(output)
    print(output / '报告.md')


if __name__ == '__main__':
    main()
