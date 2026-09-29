from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    # >1 marks a speculative verify forward: the batch carries M rows per
    # sequence (row = seq*M + m); attention uses it to pick the kernel's M
    verify_m: int = 1
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    # prefill-with-pool-prefix plan (set by prepare_prefill when cu_k > cu_q): prefix_slots
    #   = the pool slots of every sequence's cached prefix, concatenated in sequence order;
    #   prefix_plan = [(slot_off, cached, q_start, q_end)] per sequence, computed ONCE per
    #   step so no attention layer pays a GPU sync to recompute it
    prefix_slots: torch.Tensor | None = None
    prefix_plan: list | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, verify_m=1, prefix_slots=None, prefix_plan=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, verify_m, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, prefix_slots, prefix_plan)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
