"""Structured results for CFD workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from .pyfr import PyFRSimulationResult
from .trajectories import TrajectoryResult


@dataclass(frozen=True)
class TrajectoryExport:
    """Metadata describing an exported particle trajectory bundle."""

    path: Path
    num_particles: int
    steps: int
    result: TrajectoryResult


@dataclass(frozen=True)
class ModelingResult:
    """Encapsulates encoder/CNF artifacts and metrics."""

    encoder_state: Path | None
    cnf_state: Path | None
    metrics: Dict[str, float]


@dataclass(frozen=True)
class CFDSessionResult:
    """Top-level summary of a CFD workflow run."""

    simulation: PyFRSimulationResult
    trajectory_export: TrajectoryExport | None
    modeling: ModelingResult | None
    cache_dir: Path