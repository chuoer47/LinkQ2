"""Forward: token ids + paged KV -> logits.

Leaves (context, primitives, rotary, paged_decode) know nothing about
requests; qwen3 assembles them, loader only fills weights.
"""
