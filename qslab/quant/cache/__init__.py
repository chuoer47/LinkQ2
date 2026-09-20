"""Dense (non-paged) quantized KV storage: fp16 / int8 / int4 — the
round-trip accuracy oracle kept by tests/test_gpu_kernels.py.

Re-exports dropped: every caller imports kv_cache by full path.
"""
