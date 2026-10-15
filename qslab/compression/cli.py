"""Command line entry point for offline model compression."""
from __future__ import annotations

import argparse

from qslab.artifacts.io import save_checkpoint
from qslab.compression.algorithms import QUANTIZERS
from qslab.compression.config import CompressionConfig
from qslab.compression.pipeline import compress_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Compress a Qwen3 checkpoint")
    parser.add_argument("--model", required=True)
    parser.add_argument("--algorithm", choices=tuple(QUANTIZERS), default="rtn")
    parser.add_argument("--out", required=True, help="complete quantized checkpoint directory")
    parser.add_argument("--calibration", help="token id cache required by AWQ")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--group-size", type=int)
    args = parser.parse_args()
    group_size = args.group_size or QUANTIZERS[args.algorithm].default_group_size
    config = CompressionConfig(algorithm=args.algorithm, group_size=group_size,
                               calibration_path=args.calibration, device=args.device)
    artifact = compress_checkpoint(args.model, config)
    save_checkpoint(artifact, args.model, args.out)
    print(f"saved checkpoint to {args.out} ({len(artifact.weights)} quantized weights)")
