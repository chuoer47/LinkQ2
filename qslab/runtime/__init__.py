"""L3 runtime - vendored from nano-vllm (MIT), adapted for qslab.

Paged int4 KV, CUDA Graph capture, continuous batching, prefix caching,
speculative decoding.

The import graph inside this package is a DAG that only ever points down:

    engine/    request lifecycle: scheduler, draft proposers, LLMEngine
    execute/   one step: batch the rows, run the graph, sample / verify
    model/     forward: token ids + paged KV -> logits
    state/     per-sequence state + the KV block table
    config.py, sampling_params.py   the two knobs objects every layer reads

attention_store.py at the root is the one exception: zero references, pending
the TODO 18 decision.
"""
