"""Convert a portable compression artifact to a backend deployment format."""
from __future__ import annotations

import argparse

from qslab.artifacts.io import load_artifact
from qslab.conversion.export_qslab_w4 import export_qslab_w4
from qslab.conversion.nunchaku_awq import export_nunchaku_awq


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a compression artifact")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--backend", choices=("qslab-w4", "nunchaku-awq"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    artifact = load_artifact(args.artifact)
    if args.backend == "qslab-w4":
        export_qslab_w4(artifact, args.out)
    elif args.backend == "nunchaku-awq":
        export_nunchaku_awq(artifact, args.out)
