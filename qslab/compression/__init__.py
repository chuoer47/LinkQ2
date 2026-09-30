"""Offline model compression; independent of online generation and scheduling."""
from qslab.compression.config import CompressionConfig
from qslab.compression.pipeline import compress_checkpoint, quantize_model

__all__ = ["CompressionConfig", "compress_checkpoint", "quantize_model"]
