"""One isolated model measurement; launched by the benchmark runner."""
import argparse
import json
import time
from pathlib import Path

import torch

from adapters.model_builder import build_hf_causal_lm, build_quantized_causal_lm
from qslab.runtime.quantization import QuantizedWeights
from .metrics import (cleanup, decode_cycle, decode_timing, forward_timing, memory_snapshot,
                      paired_probe, probe, quality)


def storage_bytes(model):
    storages = {(str(t.device), t.untyped_storage().data_ptr()): t.untyped_storage().nbytes()
                for t in list(model.parameters()) + list(model.buffers())}
    return sum(storages.values())


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--samples', required=True)
    parser.add_argument('--mode', choices=['fp16', 'fake', 'real'], required=True)
    parser.add_argument('--quality-only', action='store_true')
    parser.add_argument('--backend', choices=['gemlite', 'marlin'], default='gemlite')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(123)
    torch.set_num_threads(4)
    device = 'cuda:0'
    cache = args.out.parent / 'autotune.json'
    if args.backend == 'gemlite':
        from qslab.kernels.gemlite.core import load_config, save_config
        if cache.exists():
            load_config(cache)
    document = json.loads(Path(args.samples).read_text())
    samples = document['samples']
    ids = samples[0]['token_ids']
    torch.cuda.init()
    cleanup(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    count = 0
    layers = []
    if args.mode == 'fp16':
        model = build_hf_causal_lm(args.model, device=device)
    else:
        weights = QuantizedWeights(args.model)
        layers = [{'name': key.removesuffix('.weight')} for key in weights.names]
        model = build_quantized_causal_lm(args.model, device=device, backend=args.backend,
                                        fake=args.mode == 'fake')
        count = len(layers) if args.mode == 'real' else 0
    cleanup(device)
    result = {'mode': args.mode, 'backend': args.backend, 'checkpoint': args.model, 'replaced_layers': count,
              'load_seconds': time.perf_counter() - start,
              'weight_storage_bytes': storage_bytes(model), 'loaded_memory': memory_snapshot(device),
              'torch': torch.__version__, 'gpu': torch.cuda.get_device_name()}
    logits, coverage = probe(model, ids[:32], device, layers)
    result['executed_layers'] = len(coverage)
    result['quality'] = quality(model, samples, device)
    if args.quality_only:
        inputs = torch.tensor([ids[:40]], device=device)
        cycle = decode_cycle(model, inputs, 32, 8, device, capture_logits=True)
        decode_logits = cycle.pop('logits')
        result['decode_check'] = cycle
        del inputs
    else:
        result['prefill'] = forward_timing(model, ids, device, [128, 512], 2, 5)
        result['decode'], decode_logits = decode_timing(model, ids, device, 128, 32, 2, 5)
    reference = args.out.parent / (Path(args.model).name + '_fake.pt')
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
    if args.mode == 'real' and args.backend == 'gemlite':
        save_config(cache)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ('mode', 'replaced_layers', 'weight_storage_bytes')}), flush=True)


if __name__ == '__main__':
    main()
