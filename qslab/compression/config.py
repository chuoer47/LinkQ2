"""Configuration for an offline compression job."""
from dataclasses import dataclass


@dataclass(slots=True)
class CompressionConfig:
    algorithm: str = "rtn"
    group_size: int = 128
    calibration_path: str | None = None
    device: str = "cuda:0"
    artifact_device: str = "cpu"
    target_suffixes: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
