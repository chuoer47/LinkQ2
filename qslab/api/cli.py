"""qslab CLI: `python -m qslab.api.cli generate --model ... [flags]`.

The flags mirror the L4 facade (qslab/api/llm.py), which mirrors
qslab.runtime.config.Config — so this is the runtime stack (M8+): packed W4
weights, the int4 paged KV pool, CUDA graphs and speculative decoding. The
frozen M0-M7 engine's own knobs (kv_mode / kv_plan) are deliberately not
surfaced here; that engine is only reachable as qslab.engine.QslabEngine.
"""
from __future__ import annotations

import argparse
import json
import sys


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="qslab", description="qslab inference CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="run one prompt through the engine")
    g.add_argument("--model", required=True, help="HF model dir")
    g.add_argument("--prompt", default=" The capital of France is")
    g.add_argument("--max-tokens", type=int, default=64)
    g.add_argument("--temperature", type=float, default=None,
                   help="omit for greedy (the measurement protocol's default)")
    g.add_argument("--device", default=None,
                   help="cuda index, e.g. cuda:2 — the runtime is single-GPU and "
                        "binds the first visible device, so this is applied as "
                        "CUDA_VISIBLE_DEVICES (must be set before torch starts)")
    g.add_argument("--seed", type=int, default=None,
                   help="seed torch before generating (temperature runs)")

    # weight + KV quantization
    g.add_argument("--w4", default=None, help="packed W4 checkpoint dir")
    g.add_argument("--w4-backend", default="w4.auto",
                   choices=["w4.auto", "w4.v1", "w4.marlin"])
    g.add_argument("--smooth-kv", default=None,
                   help="SmoothAttention calibration for the int4 KV pool")

    # speculative decoding
    g.add_argument("--spec", default=None, choices=["ngram", "lookahead", "draft"],
                   help="proposal source (ngram/lookahead are free, draft is a model)")
    g.add_argument("--spec-gamma", type=int, default=4, help="draft window")
    g.add_argument("--spec-ngram-size", type=int, default=3)
    g.add_argument("--spec-lookahead-span", type=int, default=8)
    g.add_argument("--spec-adaptive-gamma", action="store_true",
                   help="shrink the draft window when proposals keep failing "
                        "(--spec draft only: the lookups cost nothing per draft)")
    g.add_argument("--draft", default=None, help="draft model dir (with --spec draft)")
    g.add_argument("--draft-w4", default=None, help="packed W4 dir for the draft")

    # runtime sizing
    g.add_argument("--max-model-len", type=int, default=4096)
    g.add_argument("--max-num-seqs", type=int, default=None)
    g.add_argument("--gpu-memory-utilization", type=float, default=None)
    g.add_argument("--enforce-eager", action="store_true", help="skip CUDA graphs")
    g.add_argument("--compile", action="store_true", help="torch.compile the model")
    g.add_argument("--stats-only", action="store_true",
                   help="print the stats json instead of the generated text")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd != "generate":
        return 1

    if args.device is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device.rsplit(":", 1)[-1]
    if args.seed is not None:
        import torch
        torch.manual_seed(args.seed)

    from qslab.api.llm import LLM, SamplingParams
    runtime = {"max_model_len": args.max_model_len}
    for flag, key in (("max_num_seqs", "max_num_seqs"),
                      ("gpu_memory_utilization", "gpu_memory_utilization")):
        v = getattr(args, flag)
        if v is not None:
            runtime[key] = v
    if args.enforce_eager:
        runtime["enforce_eager"] = True
    if args.compile:
        runtime["compile"] = True

    llm = LLM(args.model, w4=args.w4,
              w4_backend=args.w4_backend, smooth_kv=args.smooth_kv,
              spec=args.spec, spec_gamma=args.spec_gamma,
              spec_ngram_size=args.spec_ngram_size,
              spec_lookahead_span=args.spec_lookahead_span,
              spec_adaptive_gamma=args.spec_adaptive_gamma,
              draft=args.draft, draft_w4=args.draft_w4, **runtime)
    p = {"max_tokens": args.max_tokens}
    if args.temperature is not None:
        p["temperature"] = args.temperature
    res = llm.generate(args.prompt, SamplingParams(**p))
    if args.stats_only:
        print(json.dumps(res["stats"], ensure_ascii=False))
    else:
        print(res["text"])
    if res["stats"]:
        print("stats:", json.dumps(res["stats"], ensure_ascii=False),
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
