"""Command line entry point for offline model compression."""
from __future__ import annotations

import argparse

from qslab.artifacts.io import save_artifact
from qslab.compression.config import CompressionConfig
from qslab.compression.pipeline import compress_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Compress a Qwen3 checkpoint")
    parser.add_argument("--model", required=True)
    parser.add_argument("--algorithm", choices=("rtn", "rtn_clip", "awq", "nunchaku_awq"), default="rtn")
    parser.add_argument("--out", required=True, help="portable compression artifact directory")
    parser.add_argument("--calibration", help="token id cache required by AWQ")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--group-size", type=int)
    args = parser.parse_args()
    group_size = args.group_size or (64 if args.algorithm == "nunchaku_awq" else 128)
    config = CompressionConfig(algorithm=args.algorithm, group_size=group_size,
                               calibration_path=args.calibration, device=args.device)
    artifact = compress_checkpoint(args.model, config)
    save_artifact(artifact, args.out)
    print(f"saved compression artifact to {args.out} ({len(artifact.weights)} tensors)")
