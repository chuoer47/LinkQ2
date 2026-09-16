"""qslab CLI: `python -m qslab.api.cli generate --model ... [flags]`."""
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
    g.add_argument("--device", default="cuda:0")

    # quantization flags
    g.add_argument("--w4", default=None, help="packed W4 checkpoint dir")
    g.add_argument("--w4-backend", default="w4.auto",
                   choices=["w4.auto", "w4.v1", "w4.marlin"])
    g.add_argument("--kv-mode", default="fp16", choices=["fp16", "kv8", "kv4"])
    g.add_argument("--kv-plan", default=None, help="kv4 plan json")

    # speculation flags
    g.add_argument("--draft", default=None, help="draft model dir (enables spec)")
    g.add_argument("--spec-mode", default="lookahead",
                   choices=["lookahead", "chained", "dynamic"])
    g.add_argument("--spec-gamma", type=int, default=4)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "generate":
        from qslab.api.llm import LLM, SamplingParams
        llm = LLM(args.model, device=args.device, w4=args.w4,
                  w4_backend=args.w4_backend, kv_mode=args.kv_mode,
                  kv_plan=args.kv_plan, draft=args.draft,
                  spec_mode=args.spec_mode, spec_gamma=args.spec_gamma)
        res = llm.generate(args.prompt, SamplingParams(max_tokens=args.max_tokens))
        print(res["text"])
        if res["stats"]:
            print("stats:", json.dumps(res["stats"], ensure_ascii=False),
                  file=sys.stderr)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
