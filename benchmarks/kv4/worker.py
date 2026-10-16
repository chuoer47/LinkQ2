"""One isolated fixed-capacity runtime model: FP16 weights, FP16 or INT4 KV."""
import argparse
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from transformers import AutoConfig

from qslab.runtime.model.qwen3 import Qwen3ForCausalLM
from qslab.runtime.model.loader import load_model
from qslab.runtime.model.context import set_context, reset_context
from qslab.runtime.execute.model_runner import ModelRunner
from benchmarks.w4a16.metrics import cleanup, memory_snapshot, paired_probe, summarize_nll
from .reference import FP16PagedAttention


class Session:
    def __init__(self, model_path, calibration, mode, slots=1024, block_size=256, v_group=64):
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        config.max_position_embeddings = min(config.max_position_embeddings, slots)
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.float16)
        try:
            with torch.device('cuda'):
                self.model = Qwen3ForCausalLM(config).eval()
        finally:
            torch.set_default_dtype(previous)
        load_model(self.model, model_path)
        self.mode, self.block_size = mode, block_size
        blocks = slots // block_size
        self.table = torch.arange(blocks, device='cuda', dtype=torch.int32)[None]
        self.cache_start = torch.zeros(1, device='cuda', dtype=torch.int32)
        self.pool = []
        self.layers = config.num_hidden_layers
        h, d = config.num_key_value_heads, config.head_dim
        calibration_data = torch.load(calibration, map_location='cuda', weights_only=True)
        for index, layer in enumerate(self.model.model.layers):
            attention = layer.self_attn.attn
            if mode == 'fp16':
                keys = torch.empty(blocks, block_size, h, d, device='cuda', dtype=torch.float16)
                values = torch.empty_like(keys)
                self.pool.extend([keys, values])
                attention = FP16PagedAttention(attention, keys, values, self.table, self.cache_start)
                layer.self_attn.attn = attention
                attention.set_smooth_lambda(calibration_data['lambda'][index].float())
            else:
                kq = torch.zeros(slots, h, d // 8, device='cuda', dtype=torch.uint32)
                ks = torch.empty(h, d, device='cuda', dtype=torch.float16)
                vq = torch.zeros_like(kq)
                vs = torch.zeros(slots, h, d // v_group, device='cuda', dtype=torch.float16)
                attention.k_cache, attention.v_cache = (kq, ks), (vq, vs)
                attention.block_n, attention.v_group = block_size, v_group
                self.pool.extend([kq, ks, vq, vs])
        if mode == 'kv4':
            runner = ModelRunner.__new__(ModelRunner)
            runner.config, runner.model = SimpleNamespace(hf_config=config), self.model
            runner.load_smoothing(calibration)
        self.kv_bytes = sum(t.numel() * t.element_size() for t in self.pool)
        self.counts = [0] * self.layers
        self.handles = []
        for index, layer in enumerate(self.model.model.layers):
            def hook(_module, _args, _out, key=index):
                self.counts[key] += 1
            self.handles.append(layer.self_attn.attn.register_forward_hook(hook))

    def prepare(self, ids, start, prefill):
        length = ids.numel()
        self.cache_start.fill_(start)
        positions = torch.arange(start, start + length, device='cuda', dtype=torch.int64)
        slots = positions.to(torch.int32)
        cu = torch.tensor([0, length], device='cuda', dtype=torch.int32)
        set_context(prefill, cu_seqlens_q=cu, cu_seqlens_k=cu,
                    max_seqlen_q=length, max_seqlen_k=length, slot_mapping=slots,
                    context_lens=torch.tensor([start + 1], device='cuda', dtype=torch.int32),
                    block_tables=None if prefill else self.table, verify_m=length)
        return ids, positions

    def forward(self, prepared, all_logits=False):
        ids, positions = prepared
        hidden = self.model(ids, positions)
        return self.model.compute_logits(hidden if all_logits else hidden[-1:])

    def timed(self, ids, start=0, prefill=True):
        prepared = self.prepare(ids, start, prefill)
        torch.cuda.synchronize()
        began = time.perf_counter()
        logits = self.forward(prepared)
        torch.cuda.synchronize()
        return (time.perf_counter() - began) * 1000, logits


def quality(session, samples, chunk):
    rows, captured = [], []
    for index, sample in enumerate(samples):
        ids = torch.tensor(sample['token_ids'], device='cuda', dtype=torch.int64)
        nll, start = 0.0, 0
        while start < len(sample['token_ids']) - 1:
            end = min(start + (1 if start == 0 else chunk), len(sample['token_ids']) - 1)
            logits = session.forward(session.prepare(ids[start:end], start, start == 0), all_logits=True)
            nll += F.cross_entropy(logits.float(), ids[start + 1:end + 1], reduction='sum').item()
            if index == 0 and start < 32:
                captured.append(logits[:32-start].float().cpu())
            start = end
        count = len(sample['token_ids']) - 1
        rows.append(dict(index=sample['index'], start=sample['start'], tokens=count, nll=nll))
        print(f'quality sample {index+1}/{len(samples)} nll/token={nll/count:.6f}', flush=True)
    return summarize_nll(rows), torch.cat(captured)


def performance(session, token_ids, warmup=1, repeats=3, steps=32):
    ids = torch.tensor((token_ids * 3)[:1024], device='cuda', dtype=torch.int64)
    prefill, decode = [], []
    for context in [128, 512]:
        for _ in range(warmup):
            session.timed(ids[:context])
        samples = [session.timed(ids[:context])[0] for _ in range(repeats)]
        prefill.append(dict(tokens=context, median_ms=statistics.median(samples), samples_ms=samples))
        rounds = []
        for iteration in range(warmup + repeats):
            session.forward(session.prepare(ids[:context], 0, True))
            elapsed = []
            for pos in range(context, context + steps):
                ms, _ = session.timed(ids[pos:pos+1], pos, False)
                elapsed.append(ms)
            if iteration >= warmup:
                rounds.append(elapsed)
        median = statistics.median([statistics.mean(row) for row in rounds])
        decode.append(dict(context_tokens=context, steps=steps, median_ms_per_token=median,
                           tokens_per_second=1000/median, step_samples_ms=rounds))
        print(f'performance context={context} prefill={prefill[-1]["median_ms"]:.3f}ms decode={median:.3f}ms/token', flush=True)
    return prefill, decode


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--calibration', required=True)
    parser.add_argument('--samples', required=True)
    parser.add_argument('--mode', choices=['fp16', 'kv4'], required=True)
    parser.add_argument('--v-group', type=int, default=64)
    parser.add_argument('--quality-chunk', type=int, default=32)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(2026)
    torch.set_num_threads(4)
    samples = json.loads(Path(args.samples).read_text())['samples']
    session = Session(args.model, args.calibration, args.mode, v_group=args.v_group)
    result = dict(mode=args.mode, model=args.model, weight_dtype='FP16', kv_pool_slots=1024,
                  block_size=256, v_group=args.v_group, kv_bytes=session.kv_bytes,
                  quality_chunk=args.quality_chunk, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  latency_method='synchronized_host_wall', includes_last_lm_head=True,
                  includes_context_setup=False,
                  loaded_memory=memory_snapshot('cuda'))
    result['quality'], probe = quality(session, samples, args.quality_chunk)
    reference_file = args.out.parent / 'fp16_probe.pt'
    if args.mode == 'fp16':
        full_ids = torch.tensor(samples[0]['token_ids'][:32], device='cuda', dtype=torch.int64)
        full = session.forward(session.prepare(full_ids, 0, True), all_logits=True).float().cpu()
        result['chunked_vs_full_fp16'] = paired_probe(probe, full)
        assert result['chunked_vs_full_fp16']['relative_l2'] < 0.01
        torch.save(probe, reference_file)
    else:
        result['vs_fp16_probe'] = paired_probe(probe, torch.load(reference_file, weights_only=True))
        reference_file.unlink()
    result['executed_attention_layers'] = sum(count > 0 for count in session.counts)
    assert result['executed_attention_layers'] == session.layers
    for handle in session.handles:
        handle.remove()
    torch.cuda.reset_peak_memory_stats()
    result['prefill'], result['decode'] = performance(session, samples[0]['token_ids'])
    result['measured_memory'] = memory_snapshot('cuda')
    reset_context()
    del session
    cleanup('cuda')
    result['released_memory'] = memory_snapshot('cuda')
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({'mode': args.mode, 'ppl': result['quality']['ppl'], 'kv_bytes': result['kv_bytes']}), flush=True)


if __name__ == '__main__':
    main()
