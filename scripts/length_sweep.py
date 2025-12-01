import argparse
from pathlib import Path
from typing import List

import torch

from src.modeling.datasets import CFDTrajectorySequenceDataset
from src.networks.cnf import CNFModel
from src.networks.variational_sde import VariationalSDEModel
from src.utils.paths import project_root


def path_length_torch(traj: torch.Tensor) -> float:
    diffs = traj[1:] - traj[:-1]
    return float(diffs.norm(dim=-1).sum().mean().item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep VSDE length-scale and compare path lengths to GT")
    parser.add_argument("--bundles", default=str(project_root() / "data/2d-euler-vortex/trajectories"))
    parser.add_argument("--cnf", default=str(project_root() / "cache/checkpoints/2d-euler-vortex_cnf/cnf_latest.pt"))
    parser.add_argument("--vsde", default=str(project_root() / "cache/checkpoints/2d-euler-vortex_vsde/vsde_latest.pt"))
    parser.add_argument("--scales", nargs="+", type=float, default=[0.6, 0.8, 1.0, 1.2])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    dataset = CFDTrajectorySequenceDataset([args.bundles])
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    traj_batch, times_batch, context_batch, mask_batch = next(iter(loader))
    device = args.device
    if device == "auto":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    traj_batch = traj_batch.to(device=device, dtype=torch.float32)
    times_batch = times_batch.to(device=device, dtype=torch.float32)
    context_batch = context_batch.to(device=device, dtype=torch.float32)
    mask_batch = mask_batch.to(device=device, dtype=torch.float32)

    # Ground-truth path length
    gt_len = path_length_torch(traj_batch)

    # Load models
    cnf = CNFModel(dim=2, cond_dim=3, hidden_dim=64, depth=3)
    cnf.load_state_dict(torch.load(args.cnf, map_location="cpu"))
    vsde = VariationalSDEModel(cnf, z_dim=2, ctx_dim=128, drift_hidden=256, diffusion_learnable=True).to(device).eval()
    vsde.load_state_dict(torch.load(args.vsde, map_location="cpu"), strict=False)

    print(f"GT path length (avg over batch): {gt_len:.6f}")
    results: List[tuple[float, float]] = []
    with torch.no_grad():
        for s in args.scales:
            times_out, traj_out, _u = vsde.sample_posterior(
                traj_batch, times_batch, context_batch, mask_batch, n_particles=1, n_integration_steps=60, length_scale=float(s)
            )
            vsde_traj = traj_out[:, 0, :, :]
            vsde_len = path_length_torch(vsde_traj)
            results.append((s, vsde_len))
            print(f"scale={s:.3f} | VSDE path length: {vsde_len:.6f}")

    # Simple sanity: monotonicity expectation
    monotonic = all(results[i][1] <= results[i + 1][1] for i in range(len(results) - 1))
    print(f"Monotonic increase with scale: {monotonic}")


if __name__ == "__main__":
    main()