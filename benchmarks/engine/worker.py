"""Measure the real LLMEngine, including admission, scheduling and sampling."""
import argparse
import gc
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
import torch.nn.functional as F

from qslab.api import LLM
from qslab.runtime.sampling_params import SamplingParams


def released(engine):
    manager = engine.scheduler.block_manager
    return (engine.is_finished() and not manager.used_block_ids
            and len(manager.free_block_ids) == len(manager.blocks)
            and all(block.ref_count == 0 for block in manager.blocks))


def cycle(engine, ids, tokens):
    torch.cuda.synchronize()
    started = perf_counter()
    seq = engine.add_request(ids, SamplingParams(max_tokens=tokens, temperature=0, ignore_eos=True))
    first = last = None
    while not seq.is_finished:
        engine.step()
        torch.cuda.synchronize()
        if seq.num_completion_tokens:
            last = perf_counter()
            if first is None:
                first = last
    assert seq.num_completion_tokens == tokens and released(engine)
    total = last - started
    return {'ttft_ms': (first - started) * 1000,
            'tpot_ms': (last - first) * 1000 / (tokens - 1),
            'e2e_ms': total * 1000, 'output_tokens_per_s': tokens / total,
            'token_ids': seq.completion_token_ids}


def quality(engine, reference):
    samples = torch.load(reference, map_location='cpu', weights_only=True)['samples']
    actual, expected, targets, free_matches = [], [], [], []
    for sample in samples:
        count = len(sample['token_ids'])
        free = cycle(engine, sample['prompt_ids'], count)['token_ids']
        free_matches.extend(a == b for a, b in zip(free, sample['token_ids']))
        seq = engine.add_request(sample['prompt_ids'], SamplingParams(
            max_tokens=count, temperature=0, ignore_eos=True))
        position = 0

        def force_reference(module, inputs, output):
            nonlocal position
            actual.append(inputs[0][0].detach().half().cpu())
            token = sample['token_ids'][position]
            position += 1
            return torch.tensor([token], device=output.device)

        hook = engine.model_runner.sampler.register_forward_hook(force_reference)
        try:
            while not seq.is_finished:
                engine.step()
        finally:
            hook.remove()
        assert released(engine)
        expected.extend(sample['logits'])
        targets.extend(sample['token_ids'])
    a, b = torch.stack(actual).float(), torch.stack(expected).float()
    target = torch.tensor(targets)
    finite = bool(torch.isfinite(a).all())
    return {'positions': len(targets), 'finite': finite,
            'teacher_forced_top1_agreement': float((a.argmax(-1) == b.argmax(-1)).float().mean()),
            'relative_l2': float((a - b).norm() / b.norm()),
            'mean_abs_logit_error': float((a - b).abs().mean()),
            'reference_token_nll_delta': float(F.cross_entropy(a, target) - F.cross_entropy(b, target)),
            'free_generation_token_agreement': sum(free_matches) / len(free_matches)}


def lifecycle(engine, ids):
    params = SamplingParams(max_tokens=8, temperature=0, ignore_eos=True)
    seq = engine.add_request(ids[:16], params)
    engine.abort_request(seq.seq_id)
    waiting_cancel = released(engine) and seq.finish_reason == 'cancelled'
    seq = engine.add_request(ids[:16], params)
    engine.step()
    engine.abort_request(seq.seq_id)
    running_cancel = released(engine) and seq.finish_reason == 'cancelled'
    edge = engine.scheduler.max_model_len - 2
    long_ids = (ids * ((edge + len(ids) - 1) // len(ids)))[:edge]
    seq = engine.add_request(long_ids, params)
    while not seq.is_finished:
        engine.step()
    length_stop = (len(seq) == engine.scheduler.max_model_len
                   and seq.num_completion_tokens == 2 and seq.finish_reason == 'length'
                   and released(engine))
    bad_inputs = True
    for prompt in ([], long_ids + [1, 2]):
        try:
            engine.add_request(prompt, params)
        except ValueError:
            pass
        else:
            bad_inputs = False
    return {'waiting_cancel': waiting_cancel, 'running_cancel': running_cancel,
            'context_length_stop': length_stop, 'invalid_prompt_rejected': bad_inputs,
            'all_pages_returned': released(engine)}


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--kv', choices=['fp16', 'kv4'], required=True)
    p.add_argument('--calibration', required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--tokens', type=int, default=16)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--contexts', nargs='+', type=int, default=[128, 512])
    p.add_argument('--max-model-len', type=int, default=768)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(123)
    from qslab.kernels.gemlite.core import load_config, save_config
    tuning = args.out.parent / 'autotune.json'
    if tuning.exists():
        load_config(tuning)
    ids = json.loads(args.inputs.read_text())['token_ids']
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    start = perf_counter()
    llm = LLM(args.model, kv_cache_dtype=args.kv,
        smooth_kv=args.calibration if args.kv == 'kv4' else None,
        max_num_seqs=1, max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len, num_kvcache_blocks=4,
        kvcache_block_size=256, enforce_eager=True, gpu_memory_utilization=0.9)
    engine = llm.engine
    # Timed repetitions are cold prefill, without cached-prefix work being skipped.
    engine.scheduler.block_manager.ENABLE_PREFIX_CACHE = False
    result = {'model': args.model, 'kv': args.kv, 'gpu': torch.cuda.get_device_name(),
        'torch': torch.__version__, 'load_seconds': perf_counter() - start,
        'weight_bytes': llm.weight_memory_bytes(), 'kv_bytes': llm.kv_memory_bytes(),
        'resident_allocated_bytes': torch.cuda.memory_allocated(),
        'startup_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'kv_blocks': 4, 'block_size': 256}
    result['quality'] = quality(engine, args.reference)
    # One complete warmup for every shape, then three timed complete requests.
    for context in args.contexts:
        cycle(engine, ids[:context], args.tokens)
    torch.cuda.reset_peak_memory_stats()
    result['performance'] = []
    for context in args.contexts:
        runs = [cycle(engine, ids[:context], args.tokens) for _ in range(args.repeats)]
        metrics = {key: median(row[key] for row in runs)
                   for key in ('ttft_ms', 'tpot_ms', 'e2e_ms', 'output_tokens_per_s')}
        result['performance'].append({'context': context, 'tokens': args.tokens,
            **metrics, 'runs': runs})
    result['runtime_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    result['lifecycle'] = lifecycle(engine, ids)
    result['replaced_w4_layers'] = sum(type(m).__name__ == 'GemLiteLinear'
                                      for m in engine.model_runner.model.modules())
    llm.close()
    llm.close()
    gc.collect()
    torch.cuda.empty_cache()
    result['allocated_bytes_after_close'] = torch.cuda.memory_allocated()
    result['released_all_tensors'] = torch.cuda.memory_allocated() <= before + 1024 * 1024
    result['passed'] = (result['quality']['finite'] and all(result['lifecycle'].values())
                        and result['released_all_tensors'])
    if args.kv == 'fp16':
        result['passed'] &= (result['quality']['relative_l2'] <= 0.02
                             and result['quality']['teacher_forced_top1_agreement'] >= 0.95)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    save_config(tuning)
    if not result['passed']:
        raise RuntimeError(f'acceptance failed; inspect {args.out}')


if __name__ == '__main__':
    main()
