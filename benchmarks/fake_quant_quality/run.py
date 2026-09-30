"""Run the Qwen3-8B fake quantization PPL comparison."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from benchmarks.fake_quant_quality.config import ALGORITHMS, ROOT, BenchmarkConfig
from benchmarks.fake_quant_quality.data import prepare


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--algorithms", nargs="+", choices=(*ALGORITHMS, "baseline"),
                        default=("baseline", *ALGORITHMS))
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing metrics for selected algorithms")
    args = parser.parse_args()

    config = BenchmarkConfig(
        model=(args.model or BenchmarkConfig.model),
        output=(args.output or BenchmarkConfig.output), device=args.device)
    calib_path, eval_path = prepare(config, force=args.force_data)
    config.output.mkdir(parents=True, exist_ok=True)
    worker = Path(__file__).resolve().with_name("worker.py")
    for algorithm in args.algorithms:
        result_path = config.output / f"{algorithm}.json"
        if result_path.exists() and not args.force:
            print(f"skip existing result: {result_path}")
            continue
        env = os.environ.copy()
        if args.device.startswith("cuda:"):
            env["CUDA_VISIBLE_DEVICES"] = args.device.split(":", 1)[1]
            worker_device = "cuda:0"
        else:
            worker_device = args.device
        command = [sys.executable, str(worker), "--algorithm", algorithm,
                   "--model", str(config.model),
                   "--evaluation", str(eval_path), "--output", str(result_path),
                   "--device", worker_device]
        if algorithm not in ("baseline", "rtn", "rtn_clip"):
            command.extend(("--calibration", str(calib_path)))
        subprocess.run(command, check=True, env=env, cwd=ROOT)

    results = {}
    for algorithm in args.algorithms:
        path = config.output / f"{algorithm}.json"
        if path.exists():
            results[algorithm] = json.loads(path.read_text())
    baseline = results.get("baseline", {}).get("ppl")
    summary = {"model": str(config.model), "evaluation_dataset": config.evaluation_dataset,
               "results": results, "ppl_delta_vs_baseline": {}}
    if baseline:
        summary["ppl_delta_vs_baseline"] = {
            name: {"absolute": row["ppl"] - baseline,
                  "relative_percent": (row["ppl"] / baseline - 1.0) * 100.0}
            for name, row in results.items() if name != "baseline"}
    (config.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"summary: {config.output / 'summary.json'}")


if __name__ == "__main__":
    main()
