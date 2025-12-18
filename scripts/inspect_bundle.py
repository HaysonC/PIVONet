#!/usr/bin/env python3
"""Inspect a single trajectory bundle (.npz) and report numeric health.

Usage:
  python scripts/inspect_bundle.py data/axial-compressor/trajectories/trajectories_n1024_20251201-030151.npz
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np


def inspect(path: Path, max_show: int = 3):
    with np.load(path, allow_pickle=False) as data:
        keys = list(data.keys())
        print(f"Keys: {keys}")
        history = data["history"]
        timesteps = data["timesteps"]
    print(f"history.shape = {history.shape}")
    print(f"timesteps.shape = {timesteps.shape}")
    print(f"timesteps (first 10): {timesteps[:10]}")

    # global stats
    finite_mask = np.isfinite(history)
    total = history.size
    finite = np.count_nonzero(finite_mask)
    print(f"Finite entries: {finite}/{total} ({finite / total:.2%})")
    if finite == 0:
        print(
            "No finite entries in history; cannot repair automatically. Consider regenerating bundle."
        )
        return 1

    # per-particle stats
    num_particles = history.shape[1]
    bad_particles = []
    for i in range(num_particles):
        traj = history[:, i, :]
        if not np.isfinite(traj).all():
            bad_particles.append(i)
    print(f"Bad particles: {len(bad_particles)}/{num_particles}")
    if bad_particles:
        print("First bad particle indices:", bad_particles[:max_show])
        # show sample bad particle data
        for idx in bad_particles[:max_show]:
            print(f"--- particle {idx} sample ---")
            print(history[:, idx, :])
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(2)
    sys.exit(inspect(path))
