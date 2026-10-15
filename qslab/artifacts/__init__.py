"""Complete backend-independent model checkpoints."""
from qslab.artifacts.schema import CompressionArtifact, QuantizedTensor
from qslab.artifacts.io import save_checkpoint

__all__ = ["CompressionArtifact", "QuantizedTensor", "save_checkpoint"]
