"""Dataset abstractions that adapt CFD trajectory bundles for neural training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class _ParticleRecord:
    bundle_path: Path
    index: int


def _resolve_bundle_paths(sources: Sequence[str | Path]) -> list[Path]:
    candidates: list[Path] = []
    for source in sources:
        path = Path(source)
        if path.is_file():
            candidates.append(path)
        else:
            candidates.extend(sorted(path.glob("*.npz")))
    return [p.resolve() for p in candidates if p.suffix == ".npz"]


class CFDEndpointDataset(Dataset):
    """Return final particle positions with simple context derived from CFD bundles."""

    def __init__(self, bundle_sources: Sequence[str | Path]) -> None:
        self.bundle_paths = _resolve_bundle_paths(bundle_sources)
        if not self.bundle_paths:
            raise FileNotFoundError("No CFD trajectory bundles were found.")
        self.records: list[_ParticleRecord] = []
        for path in self.bundle_paths:
            with np.load(path, allow_pickle=False) as data:
                history = data["history"]
            num_particles = history.shape[1]
            self.records.extend(_ParticleRecord(bundle_path=path, index=i) for i in range(num_particles))
        self._cache: dict[Path, dict[str, np.ndarray]] = {}

    def __len__(self) -> int:  # pragma: no cover - simple container
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        bundle = self._load_bundle(record.bundle_path)
        history = bundle["history"]
        timesteps = bundle["timesteps"]
        traj = history[:, record.index, :]  # (T, 2)
        x0 = torch.from_numpy(traj[0]).float()
        x_final = torch.from_numpy(traj[-1]).float()
        duration = float(timesteps[-1] - timesteps[0]) if len(timesteps) > 1 else float(len(traj) - 1)
        theta = torch.tensor([duration], dtype=torch.float32)
        context = torch.cat([x0, theta], dim=0)
        return x_final, x0, theta, context

    def _load_bundle(self, path: Path) -> dict[str, np.ndarray]:
        cached = self._cache.get(path)
        if cached is not None:
            return cached
        with np.load(path, allow_pickle=False) as data:
            bundle = {"history": data["history"], "timesteps": data["timesteps"]}
        self._cache[path] = bundle
        return bundle


class CFDTrajectorySequenceDataset(Dataset):
    """Return entire particle trajectories, contexts, and observation masks."""

    def __init__(self, bundle_sources: Sequence[str | Path]) -> None:
        self.bundle_paths = _resolve_bundle_paths(bundle_sources)
        if not self.bundle_paths:
            raise FileNotFoundError("No CFD trajectory bundles were found.")
        self.records: list[_ParticleRecord] = []
        for path in self.bundle_paths:
            with np.load(path, allow_pickle=False) as data:
                history = data["history"]
            num_particles = history.shape[1]
            self.records.extend(_ParticleRecord(bundle_path=path, index=i) for i in range(num_particles))
        self._cache: dict[Path, dict[str, np.ndarray]] = {}

    def __len__(self) -> int:  # pragma: no cover - simple container
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        bundle = self._load_bundle(record.bundle_path)
        history = bundle["history"]
        timesteps = bundle["timesteps"]
        traj = torch.from_numpy(history[:, record.index, :]).float()  # (T, 2)
        if len(timesteps) == traj.shape[0]:
            times = torch.from_numpy(timesteps).float()
            duration = float(timesteps[-1] - timesteps[0]) if len(timesteps) > 1 else 1.0
        else:
            duration = float(traj.shape[0] - 1)
            times = torch.linspace(0.0, duration, steps=traj.shape[0])
        times = (times - times.min()) / max(times.max() - times.min(), 1e-6)
        x0 = traj[0]
        theta = torch.tensor([float(duration)], dtype=torch.float32)
        context = torch.cat([x0, theta], dim=0)
        mask = torch.ones(traj.shape[0], dtype=torch.float32)
        return traj, times, context, mask

    def _load_bundle(self, path: Path) -> dict[str, np.ndarray]:
        cached = self._cache.get(path)
        if cached is not None:
            return cached
        with np.load(path, allow_pickle=False) as data:
            bundle = {"history": data["history"], "timesteps": data["timesteps"]}
        self._cache[path] = bundle
        return bundle
