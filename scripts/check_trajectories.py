#!/usr/bin/env python3
"""Scan trajectory bundles for non-finite values and report offending files/indices.

Usage:
  python scripts/check_trajectories.py --dir data/axial-compressor/trajectories --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def check_bundle(path: Path, verbose: bool = False):
    try:
        with np.load(path, allow_pickle=False) as data:
            history = data["history"]
    except Exception as e:
        print(f"ERROR loading {path}: {e}")
        return False

    bad = []
    num_particles = history.shape[1]
    for i in range(num_particles):
        traj = history[:, i, :]
        if not np.isfinite(traj).all():
            bad.append(i)
            if verbose:
                print(f"  bad particle {i} sample:\n{traj}")
    if bad:
        print(f"BAD: {path} -> {len(bad)} bad particles (first indices: {bad[:10]})")
        return False
    else:
        print(f"OK:  {path} ({num_particles} particles)")
        return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dir", required=True, help="Directory containing .npz trajectory bundles"
    )
    p.add_argument(
        "--verbose", action="store_true", help="Print samples for bad particles"
    )
    args = p.parse_args()

    d = Path(args.dir)
    if not d.exists():
        print(f"Directory not found: {d}")
        sys.exit(2)

    files = sorted(d.glob("*.npz"))
    if not files:
        print(f"No .npz files found in {d}")
        sys.exit(1)

    ok = True
    for f in files:
        good = check_bundle(f, verbose=args.verbose)
        ok = ok and good

    if not ok:
        print("One or more bundles contain non-finite values.")
        sys.exit(3)
    print("All bundles look finite.")
    sys.exit(0)


if __name__ == "__main__":
    main()
