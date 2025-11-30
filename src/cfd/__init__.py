"""CFD package orchestrating PyFR, trajectories, and modeling."""

from .modeling import HybridModel
from .pipeline import CFDPipeline
from .visualization import CFDVisualizer

__all__ = ["CFDPipeline", "CFDVisualizer", "HybridModel"]