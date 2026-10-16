"""Small decode attribution experiment; runtime and kernel implementations are unchanged."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import gzip
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
from time import perf_counter

from benchmarks.kv4.run import run_job


MODES = ['fp16-fp16', 'w4-fp16', 'fp16-kv4', 'w4-kv4']


def family(name):
    lower = name.lower()
    if 'gemv_revsplitk' in lower or 'gemm_splitk' in lower:
        return 'w4_linear'
    if any(s in lower for s in ['gemv', 'gemm', 'cublas']):
        return 'fp16_linear'
    if 'kv4_paged_decode' in lower:
        return 'kv4_attention'
    if 'flash' in lower:
        return 'fp16_attention'
    if 'store' in lower and ('kv' in lower or 'cache' in lower):
        return 'kv_store'
    return 'other'


def trace_summary(path):
    data = json.loads(path.read_text())
    events = data['traceEvents']
    ranges = [e for e in events if e.get('cat') == 'user_annotation'
              and e.get('name', '').startswith('diagnostic/')]
    rows = []
    for scope in ranges:
        start, end = scope['ts'], scope['ts'] + scope['dur']
        selected = [e for e in events if e.get('ph') == 'X'
                    and start <= e['ts'] and e['ts'] + e.get('dur', 0) <= end]
        kernels = [e for e in selected if e.get('cat') == 'kernel']
        activity = [e for e in selected if e.get('cat') in ['kernel', 'gpu_memcpy', 'gpu_memset']]
        if not kernels:
            raise RuntimeError('Profiler did not record GPU kernels inside ' + scope['name'])
        intervals = sorted((e['ts'], e['ts'] + e['dur']) for e in activity)
        busy, left, right = 0, *intervals[0]
        for a, b in intervals[1:]:
            if a > right:
                busy += right - left
                left, right = a, b
            else:
                right = max(right, b)
        busy += right - left
        span = max(b for _, b in intervals) - intervals[0][0]
        groups = defaultdict(lambda: {'count': 0, 'ms': 0.0})
        names = defaultdict(lambda: {'count': 0, 'ms': 0.0})
        for event in kernels:
            for target, key in [(groups, family(event['name'])), (names, event['name'])]:
                target[key]['count'] += 1
                target[key]['ms'] += event['dur'] / 1000
        runtime = [e for e in selected if e.get('cat') == 'cuda_runtime']
        rows.append({'scope': scope['name'], 'host_range_ms': scope['dur'] / 1000,
                     'kernel_count': len(kernels), 'kernel_sum_ms': sum(e['dur'] for e in kernels) / 1000,
                     'gpu_span_ms': span / 1000, 'gpu_busy_ms': busy / 1000,
                     'gpu_gap_ms': (span - busy) / 1000,
                     'runtime_calls': dict(Counter(e['name'] for e in runtime)),
                     'runtime_sum_ms': sum(e['dur'] for e in runtime) / 1000,
                     'families': dict(groups), 'kernels': dict(names)})
    with gzip.open(str(path) + '.gz', 'wt', encoding='utf-8') as out:
        json.dump(data, out)
    path.unlink()
    return rows


def timed(call, torch):
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    started = perf_counter()
    begin.record()
    result = call()
    submitted = perf_counter()
    end.record()
    end.synchronize()
    return {'wall_ms': (perf_counter() - started) * 1000,
            'host_call_ms': (submitted - started) * 1000,
            # Event elapsed includes idle gaps while the CPU submits eager work.
            'stream_elapsed_ms': begin.elapsed_time(end)}, result


def medians(rows):
    return {key: median(r[key] for r in rows) for key in rows[0] if isinstance(rows[0][key], (int, float))}


def worker(args):
    import torch
    from qslab.api import LLM
    from qslab.kernels.gemlite.core import load_config, save_config
    from qslab.runtime.model.context import reset_context, set_context
    from qslab.runtime.sampling_params import SamplingParams
    from benchmarks.engine.worker import cycle, released

    torch.set_num_threads(4)
    torch.manual_seed(123)
    if args.tuning:
        load_config(args.tuning)
    weights, kv = args.worker.split('-')
    ids = json.loads(Path(args.samples).read_text())['samples'][0]['token_ids'][:args.context]
    llm = LLM(args.w4_model if weights == 'w4' else args.model,
              kv_cache_dtype=kv, smooth_kv=args.calibration if kv == 'kv4' else None,
              max_num_seqs=1, max_model_len=768, max_num_batched_tokens=768,
              num_kvcache_blocks=4, kvcache_block_size=256, enforce_eager=False)
    engine, runner = llm.engine, llm.engine.model_runner
    engine.scheduler.block_manager.ENABLE_PREFIX_CACHE = False
    result = {'mode': args.worker, 'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'weight_bytes': llm.weight_memory_bytes(), 'kv_bytes': llm.kv_memory_bytes(),
              'engine': {}, 'static': {}}
    with torch.inference_mode():
        for eager in [True, False]:
            runner.enforce_eager = eager
            cycle(engine, ids, args.tokens)
        engine_runs = {'eager': [], 'graph': []}
        for repeat in range(args.repeats):
            order = ['eager', 'graph'] if repeat % 2 == 0 else ['graph', 'eager']
            for execution in order:
                runner.enforce_eager = execution == 'eager'
                engine_runs[execution].append(cycle(engine, ids, args.tokens))
        result['engine'] = {key: {'median': medians(rows), 'runs': rows} for key, rows in engine_runs.items()}

        # Populate the real paged cache, then hold the next token/history/addresses fixed.
        runner.enforce_eager = False
        seq = engine.add_request(ids, SamplingParams(max_tokens=args.tokens, temperature=0, ignore_eos=True))
        engine.step()
        seq.token_ids[-1] = seq.last_token = 42
        seqs, is_prefill = engine.scheduler.schedule()
        assert not is_prefill
        input_ids, positions = runner.prepare_decode(seqs)
        runner.run_model(input_ids, positions, False)
        gv = runner.graph_vars
        set_context(False, slot_mapping=gv['slot_mapping'], context_lens=gv['context_lens'],
                    block_tables=gv['block_tables'])

        def eager_forward():
            gv['outputs'].copy_(runner.model(gv['input_ids'], gv['positions']))
            return gv['outputs']

        def graph_forward():
            runner.graphs[1].replay()
            return gv['outputs']

        calls = {'eager': eager_forward, 'graph': graph_forward}
        for call in calls.values():
            for _ in range(3):
                call()
        torch.cuda.synchronize()
        static_runs = {'eager': [], 'graph': []}
        for repeat in range(args.repeats * 2):
            order = ['eager', 'graph'] if repeat % 2 == 0 else ['graph', 'eager']
            for execution in order:
                row, output = timed(calls[execution], torch)
                assert bool(torch.isfinite(output).all())
                static_runs[execution].append(row)
        result['static'] = {key: {'median': medians(rows), 'runs': rows} for key, rows in static_runs.items()}
        result['static_input'] = {'token': 42, 'position': args.context,
                                 'context_lens': gv['context_lens'].tolist(),
                                 'block_tables': gv['block_tables'].tolist(),
                                 'slot_mapping': gv['slot_mapping'].tolist()}
        path = args.out / (args.worker + '.trace.json')
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profiler:
            for execution, call in calls.items():
                for repeat in range(args.trace_steps):
                    torch.cuda.synchronize()
                    with torch.profiler.record_function(f'diagnostic/{execution}/{repeat}'):
                        call()
                        torch.cuda.synchronize()
        profiler.export_chrome_trace(str(path))
        result['trace'] = trace_summary(path)
        reset_context()
        engine.abort_all()
        result['pages_released'] = released(engine)
        result['graph_replays'] = runner.graph_replays
        # The static direct replays intentionally do not increment runner.graph_replays.
        llm.close()
        save_config(args.out / (args.worker + '.autotune.json'))
    (args.out / (args.worker + '.json')).write_text(json.dumps(result, indent=2))
    assert result['pages_released'] and result['graph_replays'] > 0


def report(out):
    rows = [json.loads((out / (mode + '.json')).read_text()) for mode in MODES]
    lines = ['# Decode CUDA Graph 归因实验', '',
             '延迟独立测量；Profiler 时间仅用于解释活动、空隙和调用数量，不能替代无 Profiler 延迟。', '',
             '| 权重 / KV | eager TPOT ms | graph TPOT ms | eager 固定 forward ms | graph 固定 forward ms |',
             '|---|---:|---:|---:|---:|']
    for row in rows:
        e, g = row['engine']['eager']['median'], row['engine']['graph']['median']
        se, sg = row['static']['eager']['median'], row['static']['graph']['median']
        lines.append(f"| {row['mode']} | {e['tpot_ms']:.3f} | {g['tpot_ms']:.3f} | {se['wall_ms']:.3f} | {sg['wall_ms']:.3f} |")
    lines += ['', '| 配置 | 执行 | kernel 数 | kernel 累计 ms | GPU 首尾跨度 ms | GPU 活动间隙 ms | host call ms（无 Profiler） |',
              '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        for execution in ['eager', 'graph']:
            trace = [r for r in row['trace'] if r['scope'].split('/')[1] == execution]
            m = medians(trace)
            host = row['static'][execution]['median']['host_call_ms']
            lines.append(f"| {row['mode']} | {execution} | {m['kernel_count']:.0f} | {m['kernel_sum_ms']:.3f} | {m['gpu_span_ms']:.3f} | {m['gpu_gap_ms']:.3f} | {host:.3f} |")
    lines += ['', '固定 forward：同一进程、权重、已填充 KV、token=42、position=512、相同地址和元数据。',
              'eager 与捕获均执行模型 forward 并复制到同一输出缓冲；不包含 lm_head、采样、调度和元数据上传。',
              '完整 TPOT 沿用引擎 benchmark；图外包含 lm_head、采样、调度和元数据上传。',
              'GPU 活动间隙 = 首个到最后一个 GPU 活动跨度减去活动区间并集，不能全部称为 CPU 计算时间。',
              '本实验不测 PPL，也不改变已有 W4 数值不确定性结论；不据此声称无损等价或硬件带宽已被直接测量。']
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description='小规模 decode Graph 时间线与权重/KV 消融实验')
    p.add_argument('--gpu', default='1')
    p.add_argument('--model', default='models/Qwen3-8B')
    p.add_argument('--w4-model', default='models/Qwen3-8B-W4-rtn')
    p.add_argument('--calibration', default='datasets/kv4/qwen3-8b.pt')
    p.add_argument('--samples', default='datasets/w4a16/samples.json')
    p.add_argument('--tuning', type=Path, help='复用已有 GemLite autotune 参数；不提供时正常预热调优')
    p.add_argument('--context', type=int, choices=[512], default=512)
    p.add_argument('--tokens', type=int, default=16)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--trace-steps', type=int, default=3)
    p.add_argument('--out', type=Path)
    p.add_argument('--worker', choices=MODES, help=argparse.SUPPRESS)
    args = p.parse_args()
    args.out = args.out or Path('results/cuda_graph_profile') / datetime.now().strftime('%Y-%m-%d_%H%M%S')
    args.out.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    metadata = {**vars(args), 'python': sys.executable,
                'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True))}
    (args.out / 'run.json').write_text(json.dumps(metadata, indent=2, default=str))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    for mode in MODES:
        command = [sys.executable, '-m', 'benchmarks.cuda_graph.profile', '--worker', mode,
                   '--model', args.model, '--w4-model', args.w4_model, '--calibration', args.calibration,
                   '--samples', args.samples, '--context', str(args.context),
                   '--tokens', str(args.tokens), '--repeats', str(args.repeats),
                   '--trace-steps', str(args.trace_steps), '--out', str(args.out)]
        if args.tuning:
            command += ['--tuning', str(args.tuning)]
        run_job(command, args.out / (mode + '.log'), env)
    report(args.out)
    print(args.out / 'report.md')


if __name__ == '__main__':
    main()
