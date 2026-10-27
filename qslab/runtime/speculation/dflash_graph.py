"""DFlash-specific graphs using the official forward and HF StaticCache."""
from dataclasses import dataclass

import torch
from transformers import StaticCache


@dataclass
class DraftSlot:
    cache: StaticCache
    graph: torch.cuda.CUDAGraph


class DFlashGraphRunner:
    def __init__(self, model, target, config, max_model_len):
        self.model, self.target, self.config = model, target, config
        self.block = config.block_size
        # Scratch keys for padded context rows never overlap visible/noise keys.
        self.capacity = max_model_len + 2*self.block
        self.rows = torch.arange(self.block, device='cuda')
        self.indices = torch.arange(self.capacity, device='cuda')
        self.ids = torch.full((1, self.block), config.dflash_config['mask_token_id'],
                              dtype=torch.long, device='cuda')
        width = len(config.dflash_config['target_layer_ids'])*config.hidden_size
        self.features = torch.zeros(1, self.block, width, dtype=torch.bfloat16, device='cuda')
        self.meta = torch.zeros(3, dtype=torch.long, device='cuda')  # cached, context end, anchor
        self.tokens = torch.empty(self.block-1, dtype=torch.long, device='cuda')
        self.slots, self.free = [], []
        self.pool = None
        self.replays = self.eager_calls = 0

    def _forward(self, cache):
        cached, length, anchor = self.meta.unbind()
        count = length-cached
        self.ids[0, 0].copy_(anchor)
        context_pos = torch.where(self.rows < count, cached+self.rows, 0)
        noise_pos = length+self.rows
        positions = torch.cat((context_pos, noise_pos))
        writes = torch.cat((torch.where(self.rows < count, cached+self.rows,
            self.capacity-self.block+self.rows), noise_pos))
        mask = (self.indices < length+self.block).view(1, 1, 1, -1)
        hidden = self.model(noise_embedding=self.target.model.embed_tokens(self.ids).to(torch.bfloat16),
            target_hidden=self.features, position_ids=positions.unsqueeze(0), attention_mask=mask,
            past_key_values=cache, cache_position=writes, use_cache=True, is_causal=False)
        return self.target.compute_logits(hidden[:, 1:].to(self.target.lm_head.weight.dtype))

    @torch.inference_mode()
    def acquire(self):
        if self.free:
            return self.free.pop()
        cache = StaticCache(config=self.config, max_cache_len=self.capacity)
        cache.early_initialization(1, self.config.num_key_value_heads,
            self.config.head_dim, torch.bfloat16, torch.device('cuda'))
        self.meta.zero_()
        for _ in range(3):
            self._forward(cache)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, self.pool):
            self.tokens.copy_(self._forward(cache).argmax(-1)[0])
        if self.pool is None:
            self.pool = graph.pool()
        slot = DraftSlot(cache, graph)
        self.slots.append(slot)
        return slot

    @torch.inference_mode()
    def run(self, slot, features, cached, length, anchor):
        count = features.shape[0]
        if count > self.block:
            # Initial prompt features have variable length; only steady decode
            # uses the fixed block-shaped graph.
            ids = torch.full_like(self.ids, self.config.dflash_config['mask_token_id'])
            ids[0, 0] = anchor
            positions = torch.arange(cached, length+self.block, device='cuda')
            mask = (self.indices < length+self.block).view(1, 1, 1, -1)
            hidden = self.model(noise_embedding=self.target.model.embed_tokens(ids).to(torch.bfloat16),
                target_hidden=features.unsqueeze(0), position_ids=positions.unsqueeze(0),
                attention_mask=mask, past_key_values=slot.cache, cache_position=positions,
                use_cache=True, is_causal=False)
            logits = self.target.compute_logits(hidden[:, 1:].to(self.target.lm_head.weight.dtype))
            self.eager_calls += 1
            return logits.argmax(-1)[0]
        self.features[0, :count].copy_(features)
        metadata = torch.tensor([cached, length, anchor], dtype=torch.long, pin_memory=True)
        self.meta.copy_(metadata, non_blocking=True)
        slot.graph.replay()
        self.replays += 1
        return self.tokens

    def release(self, slot):
        # Length/masks define validity; a new request overwrites its visible
        # prefix before reading it. Old noise and scratch keys stay masked.
        self.free.append(slot)

    def exit(self):
        self.free.clear()
        self.slots.clear()
        self.pool = None
        self.model = self.target = None
