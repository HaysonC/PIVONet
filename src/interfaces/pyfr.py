"""Interfaces for orchestrating PyFR simulations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class PyFRSimulationResult:
    """Metadata about artifacts created by a PyFR run."""

    case_dir: Path
    mesh_path: Path
    cfg_path: Path
    pyfrm_path: Path
    backend: str
    pyfrs: List[Path]
    vtus: List[Path]
    density_files: List[Path]
    velocity_files: List[Path]
    points_files: List[Path]