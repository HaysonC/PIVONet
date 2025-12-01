import torch
import numpy as np
from pathlib import Path

from src.networks.variational_sde import VariationalSDEModel
from src.networks.cnf import CNFModel
from src.utils.load_config import load_experiment_config
from src.interfaces.data_sources import build_cfd_dataset


def path_length(traj: torch.Tensor) -> float:
    # traj: (time, batch, dim)
    diffs = traj[1:] - traj[:-1]
    return float(diffs.norm(dim=-1).sum().mean().item())


def main() -> None:
    cfg = load_experiment_config("src/experiments/euler-vortex.yaml")
    dataset = build_cfd_dataset(cfg, split="test")
    dl = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    # Load VSDE from checkpoint
    vsde_ckpt = Path("cache/checkpoints/2d-euler-vortex_vsde/latest.pt")
    cnf_ckpt = Path("cache/checkpoints/2d-euler-vortex_cnf/latest.pt")
    if not vsde_ckpt.exists() or not cnf_ckpt.exists():
        print("Missing checkpoints. Expected VSDE and CNF latest.pt.")
        return

    cnf = CNFModel(z_dim=2)
    cnf.load_state_dict(torch.load(cnf_ckpt, map_location="cpu"))
    vsde = VariationalSDEModel(cnf, z_dim=2)
    vsde.load_state_dict(torch.load(vsde_ckpt, map_location="cpu"), strict=False)
    vsde.eval()

    traj_batch, times_batch, context_batch, mask_batch = next(iter(dl))
    times_out, traj_out, _ = vsde.sample_posterior(
        traj_batch.float(), times_batch.float(), context_batch.float(), mask_batch.float(), n_particles=1, n_integration_steps=50
    )
    vsde_traj = traj_out[:, 0, :, :]

    # CNF baseline path
    def simulate_cnf_baseline(cnf_model: CNFModel, z0: torch.Tensor, times: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        if times.dim() == 2:
            times = times[0]
        z = z0
        traj = [z0]
        for idx in range(times.shape[0] - 1):
            t_i = float(times[idx].item())
            dt = (times[idx + 1] - times[idx]).clamp_min(1e-5)
            drift = cnf_model.eval_field(z, ctx, t_i)
            z = z + drift * dt
            traj.append(z)
        return torch.stack(traj, dim=0)

    cnf_traj = simulate_cnf_baseline(vsde.cnf, traj_batch[:, 0, :].float(), times_out, context_batch.float())

    print(f"VSDE steps: {vsde_traj.shape[0]} | CNF steps: {cnf_traj.shape[0]}")
    print(f"VSDE path length (avg over batch): {path_length(vsde_traj):.6f}")
    print(f"CNF path length (avg over batch): {path_length(cnf_traj):.6f}")
    # Linearity quick check: ratio of net displacement to path length
    def linearity(tr: torch.Tensor) -> float:
        disp = (tr[-1] - tr[0]).norm(dim=-1).mean()
        return float(disp / (tr[1:] - tr[:-1]).norm(dim=-1).sum().mean())

    print(f"VSDE linearity ratio: {linearity(vsde_traj):.6f}")
    print(f"CNF linearity ratio: {linearity(cnf_traj):.6f}")


if __name__ == "__main__":
    main()