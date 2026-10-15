"""Validate complete checkpoint structure and standalone fake/real inference."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from safetensors import safe_open
from transformers import AutoTokenizer

from qslab.runtime.quantization import QuantizedWeights


def inventory(path):
    weights = QuantizedWeights(path)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    assert tokenizer.encode('Hello'), 'tokenizer produced no tokens'
    total = 0
    for shard in dict.fromkeys(weights.weight_map.values()):
        with safe_open(str(weights.path / shard), framework='pt') as file:
            expected = {name for name, filename in weights.weight_map.items() if filename == shard}
            assert set(file.keys()) == expected, f'index mismatch: {shard}'
            for name in file.keys():
                tensor = file.get_tensor(name)
                total += tensor.numel() * tensor.element_size()
                if name in weights.names:
                    assert str(tensor.dtype) == 'torch.int8', name
    index = weights.path / 'model.safetensors.index.json'
    if index.exists():
        assert json.loads(index.read_text())['metadata']['total_size'] == total
    return {'tensors': len(weights.weight_map), 'quantized_layers': len(weights.names),
            'shards': len(set(weights.weight_map.values())), 'tensor_bytes': total}


def main():
    parser = argparse.ArgumentParser(description='完整 checkpoint 的索引、tokenizer 与独立推理验证')
    parser.add_argument('--checkpoints', nargs='+', default=['models/Qwen3-8B-W4-rtn', 'models/Qwen3-8B-W4-awq'])
    parser.add_argument('--samples', default='datasets/w4a16/samples.json')
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--out', type=Path, default=Path('results/checkpoint_validation'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.samples, args.out / 'samples.json')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    summaries = []
    for path in args.checkpoints:
        summary = {'checkpoint': path, **inventory(path)}
        rows = {}
        for mode in ('fake', 'real'):
            name = Path(path).name + '_' + mode
            output = args.out / (name + '.json')
            cmd = [sys.executable, '-m', 'benchmarks.w4a16.worker', '--model', path,
                   '--samples', args.samples, '--mode', mode, '--quality-only', '--out', str(output)]
            print(name, flush=True)
            with (args.out / (name + '.log')).open('w') as log:
                child = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    code = child.wait()
                except KeyboardInterrupt:
                    child.terminate()
                    child.wait()
                    raise
                if code:
                    raise RuntimeError(f'{name} failed; inspect its log')
            rows[mode] = json.loads(output.read_text())
        summary.update(fake_ppl=rows['fake']['quality']['ppl'], real_ppl=rows['real']['quality']['ppl'],
                       executed_layers=rows['real']['executed_layers'],
                       real_weight_bytes=rows['real']['weight_storage_bytes'],
                       released_allocated_bytes=rows['real']['released_memory']['allocated_bytes'],
                       probe=rows['real']['vs_fake_probe'], decode=rows['real']['vs_fake_decode'])
        assert abs(summary['real_ppl']/summary['fake_ppl'] - 1) < 0.01, 'real/fake PPL differs by more than 1%'
        summaries.append(summary)
        (args.out / 'validation.json').write_text(json.dumps(summaries, indent=2))
    lines = ['# 完整 checkpoint 验证', '',
             '仅从量化模型目录加载，HF 离线模式；PPL 8×512 token，另做 32 token 上下文与 8 步 decode。', '',
             '| checkpoint | 分片 | 张量 | 量化层 | fake PPL | real PPL |',
             '|---|---:|---:|---:|---:|---:|']
    for row in summaries:
        lines.append(f"| {Path(row['checkpoint']).name} | {row['shards']} | {row['tensors']} | "
                     f"{row['executed_layers']} | {row['fake_ppl']:.6f} | {row['real_ppl']:.6f} |")
    (args.out / '报告.md').write_text('\n'.join(lines), encoding='utf-8')
    print(args.out / '报告.md')


if __name__ == '__main__':
    main()
