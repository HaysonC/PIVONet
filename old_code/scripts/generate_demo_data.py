"""Generate a small demo dataset for the Taichi viewer.

This script creates ./data/run_demo/ and writes a sequence of binary files
positions_frame_000.bin ... positions_frame_00X.bin containing float32 x,y,z triplets
for N particles across T frames. The demo creates particles moving in simple orbits.

Usage:
    python scripts/generate_demo_data.py --frames 120 --particles 10000

"""

from pathlib import Path
import numpy as np
import argparse


def generate(out_dir: Path, frames=120, N=10000):
    out_dir.mkdir(parents=True, exist_ok=True)
    # initial positions - random in a disk
    rng = np.random.RandomState(1234)
    theta = rng.rand(N) * 2 * np.pi
    r = np.sqrt(rng.rand(N)) * 0.8
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = (rng.rand(N) - 0.5) * 0.2
    pos0 = np.stack([x, y, z], axis=1).astype(np.float32)

    # Per-particle orbital speeds
    speeds = 0.2 + rng.rand(N) * 1.0
    phase = rng.rand(N) * 2 * np.pi

    for t in range(frames):
        angle = t / float(frames) * 2 * np.pi
        # make particles orbit with some radial wobble
        cur = pos0.copy()
        cos_s = np.cos(angle * speeds)
        sin_s = np.sin(angle * speeds)
        cur[:, 0] = pos0[:, 0] * cos_s - pos0[:, 1] * sin_s
        cur[:, 1] = pos0[:, 0] * sin_s + pos0[:, 1] * cos_s
        # small z oscillation
        cur[:, 2] = pos0[:, 2] + 0.05 * np.sin(angle * speeds + phase)
        path = out_dir / f"positions_frame_{t:03d}.bin"
        cur.tofile(str(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--particles", type=int, default=10000)
    parser.add_argument("--out", type=str, default="data/run_demo")
    args = parser.parse_args()
    generate(Path(args.out), frames=args.frames, N=args.particles)
