"""CLI for the qslab runtime: one `generate` verb."""
from __future__ import annotations

import argparse
import json
import sys

from qslab.api.options import add_model_options, model_options


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

    add_model_options(g)

    # runtime sizing
    g.add_argument("--max-model-len", type=int, default=4096)
    g.add_argument("--max-num-seqs", type=int, default=None)
    g.add_argument("--chunked-prefill", action='store_true')
    g.add_argument("--max-num-batched-tokens", type=int, default=None)
    g.add_argument("--gpu-memory-utilization", type=float, default=None)
    execution = g.add_mutually_exclusive_group()
    execution.add_argument("--enforce-eager", action="store_true", help="skip CUDA graphs")
    execution.add_argument("--cuda-graph", action="store_true", help="capture and replay decode")
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
    runtime['enable_chunked_prefill'] = args.chunked_prefill
    for flag, key in (("max_num_seqs", "max_num_seqs"),
                      ("max_num_batched_tokens", "max_num_batched_tokens"),
                      ("gpu_memory_utilization", "gpu_memory_utilization")):
        v = getattr(args, flag)
        if v is not None:
            runtime[key] = v
    runtime["enforce_eager"] = not args.cuda_graph
    if args.compile:
        runtime["compile"] = True

    llm = LLM(args.model, **model_options(args), **runtime)
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
    llm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
