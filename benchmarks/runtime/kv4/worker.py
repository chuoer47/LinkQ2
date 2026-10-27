import argparse, json
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from transformers import AutoConfig
from qslab.runtime.model.qwen3 import Qwen3ForCausalLM
from qslab.runtime.model.loader import load_model
from qslab.runtime.model.context import set_context, reset_context
from qslab.runtime.execute.model_runner import ModelRunner
from benchmarks.compression.weights.metrics import summarize_nll
from benchmarks.common import SAMPLES, save
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

def quality(session, samples, chunk):
    rows = []
    for index, sample in enumerate(samples):
        ids = torch.tensor(sample['token_ids'], device='cuda', dtype=torch.int64)
        nll, start = 0.0, 0
        while start < len(sample['token_ids']) - 1:
            end = min(start + (1 if start == 0 else chunk), len(sample['token_ids']) - 1)
            logits = session.forward(session.prepare(ids[start:end], start, start == 0), all_logits=True)
            nll += F.cross_entropy(logits.float(), ids[start + 1:end + 1], reduction='sum').item()
            start = end
        count = len(sample['token_ids']) - 1
        rows.append(dict(index=sample['index'], start=sample['start'], tokens=count, nll=nll))
        print(f'quality sample {index+1}/{len(samples)} nll/token={nll/count:.6f}', flush=True)
    return summarize_nll(rows)

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['fp16','kv4'],required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(4);torch.manual_seed(2026)
    session=Session('models/Qwen3-8B','datasets/kv4/qwen3-8b.pt',args.mode)
    q=quality(session,json.loads(SAMPLES.read_text())['samples'],32)
    save(args.out,{'kv':args.mode,'ppl':round(q['ppl'],6),'prediction_tokens':q['tokens'],
                   'kv_mib':round(session.kv_bytes/2**20,2)})

if __name__=='__main__':main()
