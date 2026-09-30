"""Configuration and default paths for fake quantization quality runs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "fake_quant_quality" / "qwen3-8b"
DATASETS = ROOT / "datasets" / "fake_quant_quality" / "qwen3-8b"
ALGORITHMS = ("rtn", "rtn_clip", "awq", "nunchaku_awq")


@dataclass(frozen=True)
class BenchmarkConfig:
    model: Path = ROOT / "models" / "Qwen3-8B"
    output: Path = RESULTS
    dataset_cache: Path = DATASETS
    calibration_dataset: str = "allenai/c4:en/train"
    calibration_samples: int = 128
    calibration_seq_len: int = 2048
    evaluation_dataset: str = "Salesforce/wikitext:wikitext-2-raw-v1/test"
    max_length: int = 2048
    stride: int = 512
    device: str = "cuda:0"
    dtype: str = "float16"

    def group_size(self, algorithm: str) -> int:
        return 64 if algorithm == "nunchaku_awq" else 128
