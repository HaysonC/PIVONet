"""Configuration interfaces and data structures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from typing import Any


@dataclass(frozen=True)
class SimulationConfig:
    """Runtime configuration for CFD trajectory utilities."""

    version: str
    diffusion_constant: float
    data_root: Path
    velocity_subdir: str
    density_subdir: str
    trajectory_subdir: str
    pyfr_backend: str
    pyfr_case: str | None
    pyfr_output_subdir: str
    cache_subdir: str
    trajectory_particles: int
    trajectory_steps: int | None
    trajectory_dt: float

    @property
    def velocity_dir(self) -> Path:
        return self.data_root / self.velocity_subdir

    @property
    def density_dir(self) -> Path:
        return self.data_root / self.density_subdir

    @property
    def trajectory_dir(self) -> Path:
        return self.data_root / self.trajectory_subdir

    @property
    def pyfr_output_dir(self) -> Path:
        return self.data_root / self.pyfr_output_subdir

    def __getitem__(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)
