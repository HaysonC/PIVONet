"""Helpers for listing and loading stored trajectory outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np

from ..interfaces.trajectories import TrajectoryResult
from .paths import resolve_data_path


TRAJECTORY_SUBDIR = ("cfd", "trajectories")


def trajectories_dir() -> Path:
    """Return the directory where trajectory bundles are stored."""

    path = resolve_data_path(*TRAJECTORY_SUBDIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_trajectory_files(pattern: str = "*.npz") -> List[Path]:
    """List stored trajectory bundles matching a pattern."""

    return sorted(trajectories_dir().glob(pattern))


def _coerce_path(path: str | Path) -> Path:
    target = Path(path)
    if not target.is_absolute():
        target = resolve_data_path(*target.parts)
    return target


def load_trajectory_bundle(path: str | Path) -> TrajectoryResult:
    """Load a trajectory bundle saved via the CLI."""

    target = _coerce_path(path)
    if not target.exists():
        raise FileNotFoundError(f"Trajectory file not found: {target}")

    if target.suffix == ".npz":
        data = np.load(target, allow_pickle=False)
        history = data["history"]
        timesteps = data["timesteps"] if "timesteps" in data else np.arange(history.shape[0])
    elif target.suffix == ".npy":
        history = np.load(target, allow_pickle=False)
        timesteps = np.arange(history.shape[0])
    else:
        raise ValueError(f"Unsupported trajectory format: {target.suffix}")

    return TrajectoryResult(history=history, timesteps=timesteps)


def save_trajectory_bundle(result: TrajectoryResult, path: str | Path) -> Path:
    """Persist a ``TrajectoryResult`` to disk for downstream visualization."""

    target = _coerce_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, history=result.history, timesteps=np.asarray(result.timesteps))
    return target
