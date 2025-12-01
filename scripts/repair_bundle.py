#!/usr/bin/env python3
"""Attempt to repair a trajectory bundle by interpolating NaNs per-particle.

If all entries are NaN the script will exit without writing. If some particles
are entirely NaN but others are salvageable, the script will drop fully-bad
particles by default (optionally keep them as-is).

Usage:
  python scripts/repair_bundle.py <bundle.npz> --out repaired.npz

This performs linear interpolation along the time axis for NaN gaps.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


def interp_nan(traj: np.ndarray, times: np.ndarray) -> np.ndarray:
    # traj: (T, D)
    out = traj.copy()
    for d in range(out.shape[1]):
        col = out[:, d]
        mask = np.isfinite(col)
        if mask.all():
            continue
        if not mask.any():
            # cannot repair this dimension
            continue
        good_x = times[mask]
        good_y = col[mask]
        # linear interpolate across nan gaps
        interp = np.interp(times, good_x, good_y)
        out[:, d] = interp
    return out


def repair(path: Path, out: Path, drop_fully_bad: bool = True):
    with np.load(path, allow_pickle=False) as data:
        history = data["history"].copy()
        timesteps = data["timesteps"].copy()
    T, N, D = history.shape
    finite_mask = np.isfinite(history)
    total_finite = np.count_nonzero(finite_mask)
    if total_finite == 0:
        print(f"Bundle {path} contains no finite entries; cannot repair automatically.")
        return 2

    times = timesteps if len(timesteps) == T else np.linspace(0.0, float(T - 1), num=T)

    new_history = np.empty_like(history)
    kept_indices = []
    dropped = []
    for i in range(N):
        traj = history[:, i, :]
        if not np.isfinite(traj).any():
            dropped.append(i)
            if drop_fully_bad:
                continue
            else:
                new_history[:, i, :] = traj
                kept_indices.append(i)
                continue
        # repair each dimension separately
        repaired = interp_nan(traj, times)
        new_history[:, i, :] = repaired
        kept_indices.append(i)

    if drop_fully_bad and dropped:
        print(f"Dropped {len(dropped)} fully-bad particles: first indices {dropped[:10]}")
        # compact the array to kept indices
        kept_history = new_history[:, kept_indices, :]
    else:
        kept_history = new_history
    # write repaired bundle
    np.savez_compressed(out, history=kept_history, timesteps=timesteps)
    print(f"Repaired bundle written to {out}")
    if dropped:
        print(f"Note: dropped particles indices: {dropped[:10]} (total {len(dropped)})")
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundle', help='Path to .npz bundle to repair')
    p.add_argument('--out', required=True, help='Output path for repaired bundle')
    p.add_argument('--keep-bad', action='store_true', help='Keep fully-bad particles instead of dropping')
    args = p.parse_args()

    path = Path(args.bundle)
    out = Path(args.out)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(2)
    rc = repair(path, out, drop_fully_bad=not args.keep_bad)
    sys.exit(rc)
