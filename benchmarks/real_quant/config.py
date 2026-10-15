"""Experiment matrix and the quality protocol shared with fake quantization."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models/Qwen3-8B"
DATA = ROOT / "datasets/fake_quant_quality/qwen3-8b"
ARTIFACTS = ROOT / ".cache/real_quant/artifacts"
ALGORITHMS = ("rtn", "rtn_clip", "awq", "nunchaku_awq")
BACKENDS = ("qslab_w4a16", "marlin", "nunchaku_awq")
TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                   "up_proj", "down_proj")
MAX_LENGTH = 2048
STRIDE = 512
SAMPLE_COUNT = 8
SAMPLE_LENGTH = 512
SAMPLE_SEED = 20261013
PREFILL_LENGTHS = (128, 512)
DECODE_CONTEXT = 128
DECODE_STEPS = 32


def group_size(algorithm):
    return 64 if algorithm == "nunchaku_awq" else 128


def matrix(algorithms=ALGORITHMS, backends=BACKENDS):
    return [(algorithm, backend) for algorithm in algorithms for backend in backends
            if (algorithm == "nunchaku_awq") == (backend == "nunchaku_awq")]
