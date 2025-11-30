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
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility.")
    return parser.parse_args(argv)


def _default_filename(particles: int) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return f"trajectories_n{particles}_{ts}.npz"


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    config = load_config()
    # Resolve output directory under project root
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (project_root() / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine runtime parameters, preferring CLI args then falling back to config
    particles = int(args.particles) if args.particles is not None else int(config.trajectory_particles)
    max_steps = int(args.max_steps) if args.max_steps is not None else (int(config.trajectory_steps) if config.trajectory_steps is not None else None)
    dt = float(args.dt) if args.dt is not None else float(config.trajectory_dt)

    velocity_dir = args.velocity_dir if args.velocity_dir is not None else config.velocity_dir
    source = NpyVelocityFieldSource(velocity_dir)
    simulator = ParticleTrajectorySimulator(
        velocity_source=source,
        diffusion_coefficient=getattr(config, "diffusion_constant", 0.0),
        dt=dt,
        seed=args.seed,
    )

    result = simulator.simulate(num_particles=particles, max_steps=max_steps)

    filename = _default_filename(int(args.particles))
    out_path = out_dir / filename
    np.savez(out_path, history=result.history, timesteps=np.asarray(result.timesteps))
    print(f"Saved trajectory bundle to: {out_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
