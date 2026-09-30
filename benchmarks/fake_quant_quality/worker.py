"""Single-model worker; each invocation releases its GPU state on process exit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM

from benchmarks.fake_quant_quality.config import BenchmarkConfig
from benchmarks.fake_quant_quality.evaluate import evaluate_ppl
from benchmarks.fake_quant_quality.fake_quant import apply_fake_quant


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()

    config = BenchmarkConfig(model=args.model or BenchmarkConfig.model,
                             device=args.device)
    dtype = torch.float16 if config.dtype == "float16" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        str(config.model), torch_dtype=dtype, attn_implementation="eager")
    model.to(args.device).eval()
    quantized_layers = 0
    if args.algorithm != "baseline":
        calib = torch.load(args.calibration, map_location="cpu", weights_only=False)
        quantized_layers = apply_fake_quant(
            model, args.algorithm, config.group_size(args.algorithm),
            calib["token_ids"], args.device)
    evaluation = torch.load(args.evaluation, map_location="cpu", weights_only=False)
    metrics = evaluate_ppl(model, evaluation["token_ids"], args.device,
                           config.max_length, config.stride)
    result = {
        "model": str(config.model), "algorithm": args.algorithm,
        "group_size": None if args.algorithm == "baseline" else config.group_size(args.algorithm),
        "dtype": config.dtype, "quantized_layers": quantized_layers,
        "calibration_dataset": None if args.algorithm in ("baseline", "rtn", "rtn_clip")
        else config.calibration_dataset,
        "evaluation_dataset": config.evaluation_dataset,
        "max_length": config.max_length, "stride": config.stride,
        **metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
