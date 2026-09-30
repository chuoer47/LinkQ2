"""Versioned, backend independent model compression artifacts."""
from qslab.artifacts.schema import CompressionArtifact, QuantizedTensor
from qslab.artifacts.io import load_artifact, save_artifact

__all__ = ["CompressionArtifact", "QuantizedTensor", "load_artifact", "save_artifact"]
