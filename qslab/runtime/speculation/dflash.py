"""Official block-diffusion forward driven by the runtime's commit lifecycle."""
from dataclasses import dataclass, field
import gc

import torch
from transformers import AutoModel, DynamicCache

from qslab.runtime.speculation import DraftProposal


@dataclass
class DFlashState:
    cache: DynamicCache = field(default_factory=DynamicCache)
    pending: list[torch.Tensor] = field(default_factory=list)
    length: int = 0
    cached_length: int = 0
    slot: object | None = None


class DFlashProposer:
    def __init__(self, path, config, target, *, max_model_len=4096, use_graph=True):
        # auto_map selects the checkpoint's official dflash.py implementation.
        self.model = AutoModel.from_pretrained(path, config=config, trust_remote_code=True,
            local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
        self.target = target
        self.block_size = config.block_size
        self.mask_token_id = config.dflash_config['mask_token_id']
        self.drafts: dict[int, DFlashState] = {}
        self.calls = 0
        self.graph_runner = None
        if use_graph:
            from .dflash_graph import DFlashGraphRunner
            self.graph_runner = DFlashGraphRunner(self.model, target, config, max_model_len)

    def observe(self, seq_id, start, features):
        if start == 0:
            self.drop_seqs([seq_id])
            slot = self.graph_runner.acquire() if self.graph_runner else None
            self.drafts[seq_id] = DFlashState(cache=slot.cache, slot=slot) if slot else DFlashState()
        state = self.drafts[seq_id]
        if start != state.length:
            raise RuntimeError('DFlash target feature prefix is discontinuous')
        # Graph output buffers are overwritten by the next target replay.
        state.pending.append(features.to(dtype=torch.bfloat16, copy=True))
        state.length += features.shape[0]

    @torch.inference_mode()
    def propose_batch(self, seqs, limits):
        proposals = {}
        for seq in seqs:
            count = limits[seq.seq_id]
            proposal = proposals[seq.seq_id] = DraftProposal([])
            state = self.drafts.get(seq.seq_id)
            if not count or state is None:
                continue
            if state.length != len(seq) - 1:
                raise RuntimeError('DFlash features must end immediately before the anchor token')
            features = torch.cat(state.pending)
            if self.graph_runner:
                tokens = self.graph_runner.run(state.slot, features, state.cached_length,
                                               state.length, seq.last_token)
                proposal.tokens = tokens[:count].tolist()
                state.cached_length = state.length
                state.pending.clear()
                self.calls += 1
                continue
            ids = torch.full((1, self.block_size), self.mask_token_id,
                             device='cuda', dtype=torch.long)
            ids[0, 0] = seq.last_token
            embedding = self.target.model.embed_tokens(ids).to(torch.bfloat16)
            positions = torch.arange(state.cache.get_seq_length(), state.length + self.block_size,
                                     device='cuda').unsqueeze(0)
            hidden = self.model(noise_embedding=embedding,
                target_hidden=features.unsqueeze(0), position_ids=positions,
                past_key_values=state.cache, use_cache=True, is_causal=False)
            # The cache keeps context features, never the temporary noisy block.
            state.cache.crop(state.length)
            state.cached_length = state.length
            state.pending.clear()
            logits = self.target.compute_logits(hidden[:, 1:].to(self.target.lm_head.weight.dtype))
            proposal.tokens = logits.argmax(-1)[0, :count].tolist()
            self.calls += 1
        return proposals

    def drop_seqs(self, ids):
        for seq_id in ids:
            state = self.drafts.pop(seq_id, None)
            if state is not None and state.slot is not None:
                self.graph_runner.release(state.slot)

    def exit(self):
        self.drafts.clear()
        if self.graph_runner:
            self.graph_runner.exit()
            self.graph_runner = None
        self.target = None
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()
