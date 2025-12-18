"""Summarize the synthetic bundle created by the demo experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from src.utils.paths import project_root


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/demo/demo_orchestrator_bundle.npz",
        help="Relative or absolute path to the bundle produced by prepare_demo_artifacts.",
    )
    return parser.parse_args(argv)


def _resolve_input(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    root = project_root()
    return root / path


def main() -> None:
    args = _parse_args()
    bundle_path = _resolve_input(args.input)
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Bundle not found at {bundle_path}. Run prepare_demo_artifacts first."
        )

    data = np.load(bundle_path)
    history = data["history"]
    timesteps = data["timesteps"]
    mean_position = history.mean(axis=(0, 1))
    std_position = history.std(axis=(0, 1))

    print(f"Bundle stats for {bundle_path}:")
    print(f"  history shape: {history.shape}")
    print(f"  timestep count: {timesteps.shape[0]}")
    print(f"  mean position: {mean_position}")
    print(f"  std position : {std_position}")


if __name__ == "__main__":
    main()
