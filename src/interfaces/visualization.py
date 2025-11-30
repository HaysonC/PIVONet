"""Visualization-oriented interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .trajectories import TrajectoryResult


@dataclass(frozen=True)
class PlotArtifact:
    """Metadata describing a saved visualization asset."""

    path: Path | None


class TrajectoryVisualizer(Protocol):
    """Protocol for classes capable of rendering trajectories."""

    def plot(
        self,
        trajectories: TrajectoryResult,
        output_path: Path | None = None,
        **kwargs,
    ) -> PlotArtifact:  # pragma: no cover - protocol definition
        ...
