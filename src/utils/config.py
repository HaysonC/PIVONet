"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from ..interfaces.config import SimulationConfig
from .paths import project_root


def load_config(path: str | Path | None = None) -> SimulationConfig:
    """Load application configuration from ``config.yml``.

    Args:
        path: Optional path override. Defaults to ``<project_root>/config.yml``.

    Returns:
        A :class:`~src.interfaces.config.SimulationConfig` instance populated from YAML.
    """

    config_path = Path(path) if path else project_root() / "config.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw: Mapping[str, Any] = yaml.safe_load(handle) or {}

    simulation = raw.get("simulation", {})
    diffusion = float(simulation.get("diffusion_constant", 0.0))
    trajectory_subdir = str(simulation.get("trajectory_subdir", "cfd/trajectories"))
    pyfr_backend = str(simulation.get("pyfr_backend", "metal"))
    pyfr_case = simulation.get("pyfr_case")
    pyfr_output_subdir = str(simulation.get("pyfr_output_subdir", "cfd/pyfr"))
    cache_subdir = str(simulation.get("cache_subdir", "cfd/runs"))
    particles = int(simulation.get("trajectory_particles", 256))
    steps_value = simulation.get("trajectory_steps")
    steps = int(steps_value) if steps_value is not None else None
    dt = float(simulation.get("trajectory_dt", 0.01))

    return SimulationConfig(
        version=str(raw.get("version", "0.0.0")),
        diffusion_constant=diffusion,
        data_root=project_root() / "data",
        velocity_subdir=str(simulation.get("velocity_subdir", "cfd/npy/velocity")),
        density_subdir=str(simulation.get("density_subdir", "cfd/npy/density")),
        trajectory_subdir=trajectory_subdir,
        pyfr_backend=pyfr_backend,
        pyfr_case=str(pyfr_case) if pyfr_case is not None else None,
        pyfr_output_subdir=pyfr_output_subdir,
        cache_subdir=cache_subdir,
        trajectory_particles=particles,
        trajectory_steps=steps,
        trajectory_dt=dt,
    )
