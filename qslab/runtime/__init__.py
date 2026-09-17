"""L3 runtime layer — vendored from nano-vllm (MIT), adapted for qslab.

Provides: paged KV with slot-mapped int4 quantization, CUDA Graph capture,
continuous batching scheduler, prefix caching. See docs/design-m8.md.
"""
