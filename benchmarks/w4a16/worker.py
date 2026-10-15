"""One isolated model measurement; launched by the benchmark runner."""
import argparse
import json
import time
from pathlib import Path

import torch

from adapters.model_builder import build_hf_causal_lm
from qslab.runtime.quantization import QuantizedWeights, replace_quantized_linears
from qslab.kernels.gemlite.core import load_config, save_config
from .metrics import (cleanup, decode_timing, forward_timing, memory_snapshot,
                      paired_probe, probe, quality)


@torch.no_grad()
def install_fake(model, artifact):
    weights = QuantizedWeights(artifact)
    for key in weights.names:
        data = weights.layer(key)
        values, scale = data['values'].float(), data['scale'].float()
        zero = 0 if data['zero_point'] is None else data['zero_point'].float().unsqueeze(-1)
        dense = ((values.reshape(*scale.shape, -1) - zero) * scale.unsqueeze(-1)).reshape_as(values)
        if data['input_scale'] is not None:
            dense /= data['input_scale'].float()
        model.get_submodule(key.removesuffix('.weight')).weight.copy_(dense)


def storage_bytes(model):
    storages = {(str(t.device), t.untyped_storage().data_ptr()): t.untyped_storage().nbytes()
                for t in list(model.parameters()) + list(model.buffers())}
    return sum(storages.values())


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--samples', required=True)
    parser.add_argument('--artifact')
    parser.add_argument('--mode', choices=['fp16', 'fake', 'real'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(123)
    torch.set_num_threads(4)
    device = 'cuda:0'
    cache = args.out.parent / 'autotune.json'
    if cache.exists():
        load_config(cache)
    document = json.loads(Path(args.samples).read_text())
    samples = document['samples']
    ids = samples[0]['token_ids']
    torch.cuda.init()
    cleanup(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    model = build_hf_causal_lm(args.model, device='cpu')
    count = 0
    layers = []
    if args.artifact:
        weights = QuantizedWeights(args.artifact)
        layers = [{'name': key.removesuffix('.weight')} for key in weights.names]
        if args.mode == 'fake':
            install_fake(model, args.artifact)
        else:
            count = replace_quantized_linears(model, args.artifact, device=device)
    model.to(device)
    cleanup(device)
    result = {'mode': args.mode, 'artifact': args.artifact, 'replaced_layers': count,
              'load_seconds': time.perf_counter() - start,
              'weight_storage_bytes': storage_bytes(model), 'loaded_memory': memory_snapshot(device),
              'torch': torch.__version__, 'gpu': torch.cuda.get_device_name()}
    logits, coverage = probe(model, ids[:32], device, layers)
    result['executed_layers'] = len(coverage)
    result['quality'] = quality(model, samples, device)
    result['prefill'] = forward_timing(model, ids, device, [128, 512], 2, 5)
    result['decode'], decode_logits = decode_timing(model, ids, device, 128, 32, 2, 5)
    reference = args.out.parent / (Path(args.artifact).name + '_fake.pt') if args.artifact else None
    if args.mode == 'fake':
        torch.save({'probe': logits, 'decode': decode_logits}, reference)
    elif args.mode == 'real':
        expected = torch.load(reference, weights_only=True)
        result['vs_fake_probe'] = paired_probe(logits, expected['probe'])
        result['vs_fake_decode'] = paired_probe(decode_logits, expected['decode'])
        reference.unlink()
    del model
    cleanup(device)
    result['released_memory'] = memory_snapshot(device)
    if args.mode == 'real':
        save_config(cache)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ('mode', 'replaced_layers', 'weight_storage_bytes')}), flush=True)


if __name__ == '__main__':
    main()
