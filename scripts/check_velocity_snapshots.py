#!/usr/bin/env python3
"""Scan velocity .npy snapshots for NaN/Inf values and report offending files."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np


def main(dirpath: str):
    d = Path(dirpath)
    if not d.exists():
        print(f"Directory not found: {d}")
        return 2
    files = sorted(d.glob("*.npy"))
    if not files:
        print(f"No .npy files in {d}")
        return 1
    bad = []
    for f in files:
        try:
            arr = np.load(f, allow_pickle=False)
        except Exception as e:
            print(f"ERROR loading {f}: {e}")
            bad.append((f, "load-error"))
            continue
        if not np.isfinite(arr).all():
            bad.append((f, "nan"))
    if bad:
        print("Found bad snapshots:")
        for f, reason in bad[:20]:
            print(f, reason)
        return 3
    print("All snapshots finite")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: check_velocity_snapshots.py <velocity_dir>")
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
