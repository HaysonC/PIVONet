"""Generate a lightweight synthetic trajectory bundle for orchestrator demos."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from src.utils.paths import data_root, project_root


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="data/demo/demo_orchestrator_bundle.npz",
        help="Relative or absolute output path for the synthetic bundle.",
    )
    parser.add_argument(
        "--particles", type=int, default=64, help="Number of particles to synthesize."
    )
    parser.add_argument(
        "--steps", type=int, default=40, help="Number of timesteps to synthesize."
    )
    parser.add_argument(
        "--dt", type=float, default=0.05, help="Timestep spacing for generated history."
    )
    return parser.parse_args(argv)


def _resolve_output(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    root = project_root()
    if path_like.startswith("data/"):
        return root / path
    # default to data/demo under repository root
    return data_root() / "demo" / Path(path_like).name


def main() -> None:
    args = _parse_args()
    output_path = _resolve_output(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    timesteps = np.linspace(
        0.0, args.dt * (args.steps - 1), num=args.steps, endpoint=True
    )
    history = rng.normal(size=(args.steps, args.particles, 2)).cumsum(axis=0)

    np.savez(output_path, history=history, timesteps=timesteps)
    print(f"Synthetic bundle saved to {output_path}")


if __name__ == "__main__":
    main()
