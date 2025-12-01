import sys
from pathlib import Path
import numpy as np


def path_length(traj: np.ndarray) -> float:
    # traj: (time, batch, dim)
    diffs = traj[1:] - traj[:-1]
    return float(np.linalg.norm(diffs, axis=-1).sum(axis=0).mean())


def linearity(traj: np.ndarray) -> float:
    disp = np.linalg.norm(traj[-1] - traj[0], axis=-1).mean()
    pl = np.linalg.norm(traj[1:] - traj[:-1], axis=-1).sum(axis=0).mean()
    return float(disp / max(pl, 1e-12))


def main(path: Path) -> None:
    data = np.load(path)
    if "history" in data:
        traj = data["history"]  # (time, batch, dim)
    elif "trajectories" in data:
        traj = data["trajectories"]
    else:
        print("Bundle missing 'history' or 'trajectories'")
        return
    print(f"Steps: {traj.shape[0]}, Batch: {traj.shape[1]}, Dim: {traj.shape[2]}")
    print(f"Path length (avg): {path_length(traj):.6f}")
    print(f"Linearity ratio: {linearity(traj):.6f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_bundle.py <bundle.npz>")
        sys.exit(1)
    main(Path(sys.argv[1]))