import json
from pathlib import Path

import torch

from qslab.runtime.config import Config
from qslab.runtime.state.sequence import Sequence
from qslab.runtime.model.qwen3 import Qwen3ForCausalLM
from qslab.runtime.execute.sampler import Sampler
from qslab.runtime.model.context import set_context, get_context, reset_context
from qslab.runtime.model.loader import load_model, swap_w4


class ModelRunner:

    def __init__(self, config: Config):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager

        torch.cuda.set_device(0)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float16)  # qslab: fp16 everywhere (int4 kernel is fp16)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        if config.w4:
            n = swap_w4(self.model, config.w4, backend=config.w4_backend)
            assert n > 0, f"no packed weights found under {config.w4}"
            with (Path(config.w4) / "config.json").open() as f:
                weight_config = json.load(f)
            selected_backend = ("nunchaku.awq" if weight_config.get("format_id")
                                == "nunchaku_awq_gemv_v1" else config.w4_backend)
            print(f"[w4] swapped {n} linears from {config.w4} "
                  f"(backend={selected_backend})")
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
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

    def exit(self):
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()

    def call(self, method_name, *args):
        method = getattr(self, method_name, None)
        return method(*args)

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
        """Allocate the per-layer int4 paged KV pool."""
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
        # per slot (= one token): H heads x (k_q D/8 u32 + k_s f16
        #                            + v_q D/8 u32 + v_s f16) = 2*D bytes
        slot_bytes = 2 * head_dim
        block_bytes = hf_config.num_hidden_layers * self.block_size * num_kv_heads * slot_bytes
        budget = int(total * config.gpu_memory_utilization) - used_by_others - peak
        print(f"[kvalloc] total={total/2**30:.2f}GiB util={config.gpu_memory_utilization} "
              f"used_by_others={used_by_others/2**30:.2f} peak={peak/2**30:.2f} "
              f"current={current/2**30:.2f} budget={budget/2**30:.2f}GiB "
              f"block_bytes={block_bytes/1024:.1f}KiB")
        config.num_kvcache_blocks = budget // block_bytes
        assert config.num_kvcache_blocks > 0
        nl, nb = hf_config.num_hidden_layers, config.num_kvcache_blocks
        total_slots = nb * self.block_size                     # one slot per token
        nw = head_dim // 8
        vg = config.v_group
        ng = head_dim // vg
        self.kv_cache = []
        for _ in range(nl):
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
                if getattr(module, "v_group", None) is None:
                    module.v_group = vg
                layer_id += 1
        self.load_smoothing(config.smooth_kv)

    def load_smoothing(self, path: str | None):
        """Fold the calibrated SmoothAttention factors into the pool."""
        if not path:
            return
        ck = torch.load(path, map_location="cuda", weights_only=False)
        lam, kscale = ck["lambda"].float(), ck["kscale"].float()
        assert lam.shape[0] == self.config.hf_config.num_hidden_layers, \
            "layer count mismatch"
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.set_smooth_lambda(lam[layer_id].cuda())
                module.k_cache[1].copy_(kscale[layer_id].half())
                layer_id += 1
        print(f"[smooth_kv] loaded {path}: lambda/kscale for {layer_id} layers")

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
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
            input_ids.extend(seq[start:end])
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
            # one slot plan for all 28 layers (see Context.prefix_slots)
            slots = []
            prefix_plan = []
            q_off = 0
            for seq in seqs:
                cached = seq.num_cached_tokens
                sched = seq.num_scheduled_tokens
                if sched == 0:
                    continue
                prefix_plan.append((len(slots), cached, q_off, q_off + sched))
                for j in range(cached):
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
                if bi < len(seq.block_table):
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
    def run_model_verify(self, input_ids: torch.Tensor, positions: torch.Tensor, M: int):
        """All M rows produce logits — acceptance needs every position."""
        if self.enforce_eager or input_ids.size(0) > 512:
            hidden = self.model(input_ids, positions)
            return self.model.compute_logits(hidden)
        bs = input_ids.size(0) // M
        context = get_context()
        graph = self.graphs_verify[next(x for x in self.graph_bs if x >= bs)]
        graph_vars = self.graph_vars_verify
        n = bs * M
        graph_vars["input_ids"][:n] = input_ids
        graph_vars["positions"][:n] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:n] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:n])

    @staticmethod
    def _flat_drafts(seqs: list[Sequence], M: int):
        """[bs*M] draft token per verify row, -1 where there is no draft."""
        gamma = M - 1
        flat = []
        for seq in seqs:
            d = list(seq.spec_drafts or [])[:gamma]
            flat.extend(d + [-1] * (M - len(d)))
        return torch.tensor(flat, dtype=torch.int64,
                            pin_memory=True).cuda(non_blocking=True)

    @staticmethod
    def rejection_verify(logits: torch.Tensor, temperatures: torch.Tensor,
                         drafts: torch.Tensor, draft_probs: torch.Tensor | None,
                         sampled: torch.Tensor) -> torch.Tensor:
        """Leviathan probability-ratio acceptance, applied to the sampled rows."""
        # Accept draft x with probability min(1, q(x)/p(x)); on refusal resample from
        #   normalized max(0, q - p), which leaves the target's distribution exactly
        #   unchanged.
        # A rejected row can therefore never return x, so the scheduler's walk-while-row-
        #   equals-draft prefix loop stays correct for sampled proposals as well as greedy
        #   ones, where the rule degenerates to an argmax check.
        # A deterministic proposal signals p = one-hot by passing draft_probs=None.
        # Greedy rows keep their own argmax path: near temperature 0 a near-tie would land
        #   within float error of q(x) < 1 and turn the decision into a coin flip.
        out = sampled.clone()
        has_draft = drafts >= 0
        rows = has_draft & (temperatures > 1e-3)
        if not bool(rows.any()):
            return out
        t = temperatures[rows].unsqueeze(1)
        q = torch.softmax(logits[rows].float().div_(t), dim=-1)
        x = drafts[rows]
        one_hot = draft_probs is None
        if one_hot:
            p_x = torch.ones_like(x, dtype=torch.float32)
            residual = q.clone()
        else:
            p = draft_probs[rows].float()
            p_x = p.gather(1, x.unsqueeze(1)).squeeze(1)
            residual = (q - p).clamp_min(0)
        q_x = q.gather(1, x.unsqueeze(1)).squeeze(1)
        accept = torch.rand(q_x.shape, device=q.device) < (q_x / p_x.clamp_min(1e-12)).clamp(max=1.0)
        if one_hot:
            # max(0, q - delta_x): every entry of q except x's own
            residual.scatter_(1, x.unsqueeze(1), 0.0)
        total = residual.sum(1, keepdim=True)
        residual = torch.where(total > 0, residual / total.clamp_min(1e-12), q)
        resampled = torch.multinomial(residual, 1).squeeze(1)
        idx = rows.nonzero(as_tuple=True)[0]
        out[idx[accept]] = x[accept]
        out[idx[~accept]] = resampled[~accept]
        return out

    def run_verify(self, seqs: list[Sequence],
                   draft_probs: torch.Tensor | None = None) -> list[int]:
        """One verify forward; returns the committed token per row (bs*M)."""
        gamma = self.config.spec_num_drafts
        M = gamma + 1
        input_ids, positions = self.prepare_verify(seqs)
        temperatures = self.prepare_sample(seqs).repeat_interleave(M)
        logits = self.run_model_verify(input_ids, positions, M)
        token_ids = self.sampler(logits, temperatures)
        drafts = self._flat_drafts(seqs, M)
        token_ids = self.rejection_verify(logits, temperatures, drafts,
                                          draft_probs, token_ids).tolist()
        reset_context()
        return token_ids

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            hidden = self.model(input_ids, positions)
            if is_prefill:
                # only the last token of each scheduled sequence produces logits
                # (sampling happens at sequence boundaries); gathering all
                # positions would break the per-seq temperature broadcast
                cu = get_context().cu_seqlens_q
                last_idx = (cu[1:] - 1) if cu is not None else torch.tensor(
                    [hidden.size(0) - 1], device=hidden.device)
                hidden = hidden[last_idx]
            return self.model.compute_logits(hidden)
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs)
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist()
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

        if config.spec_method:
            # second family, keyed by bs at fixed M = gamma+1 (vLLM V1:
            # pad proposals to gamma — padding is neutral because the
            # acceptance loop never looks past the real draft count)
            M = config.spec_num_drafts + 1
            v_input_ids = torch.zeros(max_bs * M, dtype=torch.int64)
            v_positions = torch.zeros(max_bs * M, dtype=torch.int64)
            v_slot_mapping = torch.zeros(max_bs * M, dtype=torch.int32)
            v_context_lens = torch.zeros(max_bs, dtype=torch.int32)
            v_block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
            v_outputs = torch.zeros(max_bs * M, hf_config.hidden_size)
            self.graphs_verify = {}
            for bs in reversed(self.graph_bs):
                graph = torch.cuda.CUDAGraph()
                set_context(False, slot_mapping=v_slot_mapping[:bs * M],
                            context_lens=v_context_lens[:bs],
                            block_tables=v_block_tables[:bs], verify_m=M)
                v_outputs[:bs * M] = self.model(v_input_ids[:bs * M], v_positions[:bs * M])    # warmup
                with torch.cuda.graph(graph, self.graph_pool):
                    v_outputs[:bs * M] = self.model(v_input_ids[:bs * M], v_positions[:bs * M])
                self.graphs_verify[bs] = graph
                torch.cuda.synchronize()
                reset_context()
            self.graph_vars_verify = dict(
                input_ids=v_input_ids,
                positions=v_positions,
                slot_mapping=v_slot_mapping,
                context_lens=v_context_lens,
                block_tables=v_block_tables,
                outputs=v_outputs,
            )
