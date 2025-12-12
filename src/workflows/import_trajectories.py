"""Simulate particle trajectories from existing velocity snapshots and save as bundles.

This workflow script mirrors the CLI `import` behavior but is intended for
non-interactive orchestrated runs. It accepts simple flags and writes a
`.npz` bundle into the requested output directory.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

from src.cfd.particle_trajectories import ParticleTrajectorySimulator
from src.interfaces.data_sources import NpyVelocityFieldSource
from src.utils.config import load_config
from src.utils.paths import project_root


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=None, help="Number of particles to simulate. Falls back to config value if omitted.")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum number of velocity snapshots to consume. Falls back to config value if omitted.")
    parser.add_argument("--dt", type=float, default=None, help="Fallback timestep when snapshot timing is not respected. Falls back to config value if omitted.")
    parser.add_argument("--output-dir", default="data/double-mach/trajectories", help="Directory to write trajectory bundles into.")
    parser.add_argument("--velocity-dir", default=None, help="Optional path to the directory containing velocity .npy snapshots (overrides config).")
    parser.add_argument("--mesh-path", default=None, help="Optional explicit path to mesh point coordinates (.npy). Defaults to autodiscovery near the velocity directory.")
    parser.add_argument("--system", default=None, help="Name/slug of the CFD system to store inside the bundle metadata (defaults to inferred directory name).")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility.")
    return parser.parse_args(argv)


def _default_filename(particles: int) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return f"trajectories_n{particles}_{ts}.npz"


def _resolve_project_path(path_like: str | Path | None) -> Path | None:
    if path_like is None:
        return None
    path = Path(path_like)
    if not path.is_absolute():
        path = (project_root() / path).resolve()
    return path


def _discover_mesh_points(explicit: Path | None, velocity_dir: Path) -> np.ndarray | None:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    search_roots = [velocity_dir, velocity_dir.parent]
    for root in search_roots:
        for name in ("points.npy", "mesh_points.npy"):
            candidate = root / name
            if candidate not in candidates:
                candidates.append(candidate)
    for candidate in candidates:
        if candidate is None or not candidate.exists():
            continue
        try:
            data = np.load(candidate, allow_pickle=False)
        except Exception:
            continue
        if data.ndim == 2 and data.shape[1] >= 2:
            return np.asarray(data[:, :2], dtype=np.float32)
    return None


def _infer_system_slug(velocity_dir: Path) -> str:
    for candidate in (velocity_dir.name, velocity_dir.parent.name if velocity_dir.parent != velocity_dir else ""):
        trimmed = candidate.strip()
        if trimmed:
            return trimmed
    return "unknown-system"


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    config = load_config()
    # Resolve output directory under project root
    out_dir = _resolve_project_path(args.output_dir)
    if out_dir is None:
        raise ValueError("Invalid output directory specified.")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine runtime parameters, preferring CLI args then falling back to config
    particles = int(args.particles) if args.particles is not None else int(config.trajectory_particles)
    max_steps = int(args.max_steps) if args.max_steps is not None else (int(config.trajectory_steps) if config.trajectory_steps is not None else None)
    dt = float(args.dt) if args.dt is not None else float(config.trajectory_dt)

    velocity_override = _resolve_project_path(args.velocity_dir)
    velocity_dir = velocity_override if velocity_override is not None else config.velocity_dir
    source = NpyVelocityFieldSource(velocity_dir)
    simulator = ParticleTrajectorySimulator(
        velocity_source=source,
        diffusion_coefficient=getattr(config, "diffusion_constant", 0.0),
        dt=dt,
        seed=args.seed,
    )

    mesh_override = _resolve_project_path(args.mesh_path)
    mesh_points = _discover_mesh_points(mesh_override, velocity_dir)
    system_slug = args.system or _infer_system_slug(velocity_dir)
    if mesh_points is None:
        print("Warning: mesh points could not be resolved; bundles will omit mesh metadata.")

    result = simulator.simulate(num_particles=particles, max_steps=max_steps)

    filename = _default_filename(particles)
    out_path = out_dir / filename
    payload: dict[str, np.ndarray] = {
        "history": np.asarray(result.history, dtype=np.float32),
        "timesteps": np.asarray(result.timesteps, dtype=np.float32),
        "system": np.asarray(system_slug),
    }
    if mesh_points is not None:
        payload["mesh_points"] = mesh_points
    np.savez(out_path, allow_pickle=False, **payload)
    print(f"Saved trajectory bundle to: {out_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
