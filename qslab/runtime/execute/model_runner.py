import torch
import gc
from copy import copy
from accelerate import init_empty_weights

from qslab.runtime.config import Config
from qslab.runtime.state.sequence import Sequence
from qslab.runtime.model.qwen3 import Qwen3ForCausalLM
from qslab.runtime.execute.sampler import Sampler
from qslab.runtime.execute.cuda_graph import CUDAGraphs
from qslab.runtime.model.context import set_context, get_context, reset_context
from qslab.runtime.model.loader import load_model
from qslab.runtime.quantization import load_checkpoint_weights


class ModelRunner:

    def __init__(self, config: Config, *, feature_layers=()):
        self.config = config
        self.feature_layers = tuple(feature_layers)
        self.context_features = None
        self.feature_ranges = []
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.graph_replays = 0
        self.decode_graph_replays = 0
        self.verify_graph_replays = 0
        self.varlen_verify_calls = 0

        torch.cuda.set_device(0)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float16)  # qslab: fp16 everywhere (int4 kernel is fp16)
        torch.set_default_device("cuda")
        try:
            if getattr(hf_config, "quantization_config", None):
                # Parameters start on meta; RoPE buffers are built on the GPU.
                # Accelerate's registration hook cannot re-register our embedding
                # Parameter with its weight_loader attribute. Tie after loading.
                construction_config = copy(hf_config)
                construction_config.tie_word_embeddings = False
                with init_empty_weights():
                    self.model = Qwen3ForCausalLM(construction_config)
                n = load_checkpoint_weights(self.model, config.model,
                    backend=config.quant_backend, device="cuda:0", dtype=torch.float16)
                if hf_config.tie_word_embeddings:
                    self.model.lm_head.weight = self.model.model.embed_tokens.weight
                print(f"[quantization] loaded {n} projections ({config.quant_backend})")
            else:
                self.model = Qwen3ForCausalLM(hf_config)
                load_model(self.model, config.model)
            for module in self.model.modules():
                if hasattr(module, 'kv_cache_dtype'):
                    module.kv_cache_dtype = config.kv_cache_dtype
            self.model.eval()
            self.model.model.feature_layers = self.feature_layers
            self._initialize_execution(config)
        finally:
            torch.set_default_device("cpu")
            torch.set_default_dtype(default_dtype)

    def _initialize_execution(self, config):
        if config.compile:
            # kernel-fusion lever for the small-model draft: inductor fuses the ~1500 per-
            #   step elementwise/copy kernels; attention is dynamo-disabled so the global
            #   Context never touches the compiled regions. Compiled during warmup, then
            #   recorded into the CUDA graphs like any other kernels.
            self.model = torch.compile(self.model, dynamic=True,
                                       mode=config.compile_mode or None)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.cuda_graphs = CUDAGraphs(config, self.feature_layers)
            self.cuda_graphs.capture(self._forward, lambda: self.context_features)

    def exit(self):
        reset_context()
        self.context_features = None
        self.feature_ranges.clear()
        for name in ('cuda_graphs', 'kv_cache', 'model', 'sampler'):
            if hasattr(self, name):
                delattr(self, name)
        torch.cuda.synchronize()
        # Parameters carry bound weight_loader methods, so dropping the model
        # alone leaves Python reference cycles until the next GC collection.
        gc.collect()
        torch.cuda.empty_cache()

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """Allocate the configured paged pool, accounting for its actual layout."""
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        # memory held by OTHER processes on this GPU: mem_get_info is
        # device-wide, so a shared card would otherwise make the budget
        # negative. On an exclusive card this term is 0 and the formula
        # reduces to nano-vllm's original.
        used_by_others = (total - free) - current
        num_kv_heads = hf_config.num_key_value_heads
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        mode = config.kv_cache_dtype
        slot_bytes = 4 * head_dim if mode == 'fp16' else head_dim + 2 * (head_dim // config.v_group)
        static_bytes = 0 if mode == 'fp16' else hf_config.num_hidden_layers * num_kv_heads * head_dim * 2
        block_bytes = hf_config.num_hidden_layers * self.block_size * num_kv_heads * slot_bytes
        budget = int(total * config.gpu_memory_utilization) - used_by_others - peak - static_bytes
        print(f"[kvalloc] total={total/2**30:.2f}GiB util={config.gpu_memory_utilization} "
              f"used_by_others={used_by_others/2**30:.2f} peak={peak/2**30:.2f} "
              f"current={current/2**30:.2f} budget={budget/2**30:.2f}GiB "
              f"block_bytes={block_bytes/1024:.1f}KiB")
        capacity = budget // block_bytes
        if config.num_kvcache_blocks > 0:
            if config.num_kvcache_blocks > capacity:
                raise ValueError('requested KV pool exceeds available memory budget')
        else:
            max_blocks = config.max_num_seqs * ((config.max_model_len + self.block_size - 1) // self.block_size)
            config.num_kvcache_blocks = min(capacity, max_blocks)
        if config.num_kvcache_blocks <= 0:
            raise ValueError('no memory available for KV blocks')
        nl, nb = hf_config.num_hidden_layers, config.num_kvcache_blocks
        total_slots = nb * self.block_size                     # one slot per token
        nw = head_dim // 8
        vg = config.v_group
        ng = head_dim // vg
        self.kv_cache = []
        for _ in range(nl):
            if mode == 'fp16':
                shape = (nb, self.block_size, num_kv_heads, head_dim)
                self.kv_cache.append((torch.zeros(shape, dtype=torch.float16, device='cuda'),
                                      torch.zeros(shape, dtype=torch.float16, device='cuda')))
                continue
            # K: int4 + a STATIC per-channel scale table shared by all slots
            kq = torch.zeros(total_slots, num_kv_heads, nw,
                             dtype=torch.uint32, device="cuda")
            ks = torch.zeros(num_kv_heads, head_dim,
                             dtype=torch.float16, device="cuda")
            # V: int4 + a dynamic per-token, per-group scale
            vq = torch.zeros(total_slots, num_kv_heads, nw,
                             dtype=torch.uint32, device="cuda")
            vs = torch.zeros(total_slots, num_kv_heads, ng,
                             dtype=torch.float16, device="cuda")
            self.kv_cache.append(((kq, ks), (vq, vs)))
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[layer_id][0]
                module.v_cache = self.kv_cache[layer_id][1]
                module.v_group = vg
                module.block_n = self.block_size
                module.kv_cache_dtype = mode
                layer_id += 1
        if config.smooth_kv:
            self.load_smoothing(config.smooth_kv)

    def load_smoothing(self, path: str | None):
        """Fold the calibrated SmoothAttention factors into the pool."""
        ck = torch.load(path, map_location="cuda", weights_only=True)
        lam, kscale = ck["lambda"].float(), ck["kscale"].float()
        cfg = self.config.hf_config
        head_dim = getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads)
        expected = (cfg.num_hidden_layers, cfg.num_key_value_heads, head_dim)
        for name, tensor in [('lambda', lam), ('kscale', kscale)]:
            stored = tensor.half()
            if tensor.shape != expected or not torch.isfinite(stored).all() or (stored <= 0).any():
                raise ValueError(f'{name} must have shape {expected} and finite positive FP16 values')
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.set_smooth_lambda(lam[layer_id].cuda())
                if module.kv_cache_dtype == 'kv4':
                    module.k_cache[1].copy_(kscale[layer_id].half())
                layer_id += 1
        print(f"[smooth_kv] loaded {path}: lambda/kscale for {layer_id} layers")

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        self._record_feature_ranges(seqs)
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end] if seq.is_prefill
                             else [seq.last_token] + seq.spec_drafts)
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        prefix_slots = None
        prefix_plan = None
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache / chunked
            block_tables = self.prepare_block_tables(seqs)
            if self.config.kv_cache_dtype == 'kv4':
                # one slot plan for all 28 layers (see Context.prefix_slots)
                slots = []
                prefix_plan = []
                q_off = 0
                for seq in seqs:
                    cached = seq.num_cached_tokens
                    sched = seq.num_scheduled_tokens
                    if sched == 0:
                        continue
                    stored = cached if seq.is_prefill else cached + sched
                    prefix_plan.append((len(slots), stored, q_off, q_off + sched, seq.is_prefill))
                    for j in range(stored):
                        slots.append(seq.block_table[j // self.block_size] * self.block_size
                                     + j % self.block_size)
                    q_off += sched
                prefix_slots = torch.tensor(slots, dtype=torch.int64,
                                            pin_memory=True).cuda(non_blocking=True)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                    slot_mapping, None, block_tables,
                    prefix_slots=prefix_slots, prefix_plan=prefix_plan or None)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        self._record_feature_ranges(seqs, 1)
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            # positional: the block holding position len(seq)-1. Exact when
            # the table is canonical and correct when a verify step left it
            # longer (block_table[-1] would then point past this token)
            slot_mapping.append(seq.block_table[seq.num_blocks - 1] * self.block_size
                                + seq.last_block_num_tokens - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_verify(self, seqs: list[Sequence]):
        """Build the M-row verify batch: [last committed token, drafts...]."""
        gamma = self.config.spec_num_drafts
        M = gamma + 1
        self._record_feature_ranges(seqs, M)
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            L = len(seq)
            d = list(seq.spec_drafts or [])[:gamma]
            input_ids.extend([seq.last_token] + d + [0] * (gamma - len(d)))
            positions.extend(range(L - 1, L + gamma))
            context_lens.append(L)
            for m in range(M):
                pos = L - 1 + m
                bi = pos // self.block_size
                if m < seq.num_scheduled_tokens and bi < len(seq.block_table):
                    slot_mapping.append(seq.block_table[bi] * self.block_size + pos % self.block_size)
                else:
                    slot_mapping.append(-1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens,
                    block_tables=block_tables, verify_m=M)
        return input_ids, positions

    @torch.inference_mode()
    def verify_logits(self, seqs: list[Sequence], has_prefill: bool):
        """One forward for a mixed batch; expose only valid sampling/verify rows.

        Uniform verify uses the captured M=gamma+1 shape. Varlen uses actual
        query lengths, including near the context limit where padding would
        extend past the page table. No acceptance algorithm lives here.
        """
        M = self.config.spec_num_drafts + 1
        varlen = has_prefill or any(len(seq) + M - 1 > self.config.max_model_len for seq in seqs)
        if varlen:
            input_ids, positions = self.prepare_prefill(seqs)
            hidden = self.run_model(input_ids, positions, True)
            self.varlen_verify_calls += 1
        else:
            input_ids, positions = self.prepare_verify(seqs)
            hidden = self.run_model(input_ids, positions, False, rows=M)
        indices, counts, offset = [], [], 0
        for seq in seqs:
            count = seq.num_scheduled_tokens
            if seq.spec_verify:
                indices.extend(range(offset, offset + count))
                counts.append(count)
            elif seq.should_sample:
                indices.append(offset + (count - 1 if varlen else 0))
                counts.append(1)
            else:
                counts.append(0)
            offset += count if varlen else M
        index = torch.tensor(indices, device=hidden.device, dtype=torch.int64)
        logits = self.model.compute_logits(hidden.index_select(0, index))
        reset_context()
        return logits, counts

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool, *, rows=1):
        if is_prefill or self.enforce_eager or input_ids.size(0) // rows > max(self.cuda_graphs.batch_sizes):
            return self._forward(input_ids, positions)
        hidden, features = self.cuda_graphs.replay(input_ids, positions, get_context(), rows)
        if self.feature_layers:
            self.context_features = features
        self.graph_replays += 1
        if rows == 1:
            self.decode_graph_replays += 1
        else:
            self.verify_graph_replays += 1
        return hidden

    def _forward(self, input_ids, positions):
        hidden = self.model(input_ids, positions)
        if self.feature_layers:
            self.context_features = self.model.model.context_features
        return hidden

    def _record_feature_ranges(self, seqs, rows_per_seq=None):
        if not self.feature_layers:
            return
        self.feature_ranges = []
        offset = 0
        for seq in seqs:
            start = seq.num_cached_tokens if seq.is_prefill else len(seq) - 1
            count = seq.num_scheduled_tokens
            self.feature_ranges.append((seq.seq_id, start, offset, count))
            offset += rows_per_seq or count

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        hidden = self.run_model(input_ids, positions, is_prefill)
        sample_seqs = [seq for seq in seqs if seq.should_sample]
        token_ids = [0] * len(seqs)
        if sample_seqs:
            if is_prefill:
                cu = get_context().cu_seqlens_q
                flags = torch.tensor([seq.should_sample for seq in seqs], device=cu.device, dtype=torch.bool)
                indices = torch.nonzero(flags, as_tuple=True)[0]
                hidden = hidden[(cu[indices + 1] - 1).long()]
            logits = self.model.compute_logits(hidden)
            sampled = iter(self.sampler(logits, self.prepare_sample(sample_seqs)).tolist())
            token_ids = [next(sampled) if seq.should_sample else 0 for seq in seqs]
        reset_context()
        return token_ids
