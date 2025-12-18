"""Interfaces for loading CFD velocity fields stored in NumPy format."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence

import numpy as np



@dataclass(frozen=True)
class VelocitySnapshot:
    """Metadata for a velocity field snapshot stored as an ``.npy`` file."""

    timestep: float
    path: Path


class VelocityFieldSource:
    """Abstract interface for retrieving velocity fields over time."""

    def available_timesteps(
        self,
    ) -> Sequence[float]:  # pragma: no cover - protocol-like behavior
        raise NotImplementedError

    def iter_velocity_fields(
        self,
    ) -> Iterator[tuple[float, np.ndarray]]:  # pragma: no cover
        raise NotImplementedError


class NpyVelocityFieldSource(VelocityFieldSource):
    """Velocity field loader that streams ``.npy`` snapshots"""

    def __init__(self, velocity_dir: str | Path):
        self._velocity_dir = Path(velocity_dir)
        self._snapshots = self._discover_snapshots()
        if not self._snapshots:
            raise FileNotFoundError(
                f"No velocity snapshots found under {self._velocity_dir}. Did you export the VTU data?"
            )

    def available_timesteps(self) -> Sequence[float]:
        return [snap.timestep for snap in self._snapshots]

    def iter_velocity_fields(self) -> Iterator[tuple[float, np.ndarray]]:
        for snap in self._snapshots:
            yield snap.timestep, np.load(snap.path, allow_pickle=False)

    def _discover_snapshots(self) -> List[VelocitySnapshot]:
        if not self._velocity_dir.exists():
            return []
        pattern = re.compile(r"([0-9]+(?:\.[0-9]+)?)")
        snapshots: List[VelocitySnapshot] = []
        for npy_file in sorted(self._velocity_dir.glob("*.npy")):
            matches = pattern.findall(npy_file.stem)
            timestep = float(matches[-1]) if matches else float(len(snapshots))
            snapshots.append(VelocitySnapshot(timestep=timestep, path=npy_file))
        snapshots.sort(key=lambda snap: snap.timestep)
        return snapshots
