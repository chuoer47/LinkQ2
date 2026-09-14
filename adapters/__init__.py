"""adapters — glue to transformers/HF ecosystems.

This is the ONLY layer allowed to import transformers/vllm.
qslab core, kernels, quantizer must stay independent of them.
"""
