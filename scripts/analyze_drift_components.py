"""Analyze VSDE posterior drift components to verify 2D learning."""

import sys
from pathlib import Path
import numpy as np
import torch
from src.networks.variational_sde import VariationalSDEModel
from src.networks.cnf import CNFModel


def analyze_drift(vsde_checkpoint: Path, bundle_path: Path, device: str = "cpu"):
    """Load VSDE and analyze posterior drift components."""

    # Load checkpoint (flat state dict from VSDE)
    state_dict = torch.load(vsde_checkpoint, map_location=device)

    # Hardcode known dimensions from training
    # CNF input is 6 = 2 (z_dim) + 3 (cond_dim) + 1 (time)
    cnf = CNFModel(dim=2, cond_dim=3, hidden_dim=64, depth=3)
    vsde = VariationalSDEModel(
        cnf=cnf,
        z_dim=2,
        ctx_dim=128,
        drift_hidden=256,  # We trained with this
        diffusion_learnable=True,
    )
    vsde.load_state_dict(state_dict)
    vsde.to(device).eval()

    # Load one batch of data
    bundle = np.load(bundle_path)
    trajectories = bundle["trajectories"][:, :16]  # (T, 16, 2)
    context = bundle["context"][:16]  # (16, ctx_dim)
    t_obs = bundle.get("t_obs", np.linspace(0, 1, trajectories.shape[0]))

    # Convert to torch
    z0 = torch.tensor(trajectories[0], device=device, dtype=torch.float32)
    ctx = torch.tensor(context, device=device, dtype=torch.float32)
    t_obs_tensor = torch.tensor(t_obs, device=device, dtype=torch.float32)

    # Sample drift at various timepoints
    print("\n=== VSDE Posterior Drift Analysis ===\n")
    print(f"Analyzing {z0.shape[0]} trajectories")
    print(
        f"Initial positions range: x=[{z0[:, 0].min():.3f}, {z0[:, 0].max():.3f}], "
        f"y=[{z0[:, 1].min():.3f}, {z0[:, 1].max():.3f}]"
    )

    with torch.no_grad():
        # Encode context
        ctx_enc = vsde.context_encoder(ctx)

        # Check drift at t=0, 0.5, 1.0
        for t_val in [0.0, 0.5, 1.0]:
            t = torch.full((z0.shape[0],), t_val, device=device)

            # CNF drift
            cnf_drift = vsde.cnf.eval_field(z0, t, ctx_enc)

            # VSDE posterior drift
            posterior_drift = vsde.posterior_drift(z0, t, ctx_enc)

            # Statistics
            print(f"\n--- t = {t_val:.1f} ---")
            print(
                f"CNF drift:       x_mean={cnf_drift[:, 0].mean():.6f}, x_std={cnf_drift[:, 0].std():.6f}, "
                f"y_mean={cnf_drift[:, 1].mean():.6f}, y_std={cnf_drift[:, 1].std():.6f}"
            )
            print(f"                 magnitude_mean={cnf_drift.norm(dim=1).mean():.6f}")
            print(
                f"Posterior drift: x_mean={posterior_drift[:, 0].mean():.6f}, x_std={posterior_drift[:, 0].std():.6f}, "
                f"y_mean={posterior_drift[:, 1].mean():.6f}, y_std={posterior_drift[:, 1].std():.6f}"
            )
            print(
                f"                 magnitude_mean={posterior_drift.norm(dim=1).mean():.6f}"
            )

            # Check if y-component is non-trivial
            y_ratio = posterior_drift[:, 1].abs().mean() / (
                posterior_drift[:, 0].abs().mean() + 1e-12
            )
            print(f"Y/X ratio: {y_ratio:.4f} (>0.1 indicates significant 2D learning)")


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python scripts/analyze_drift_components.py <vsde_checkpoint> <bundle.npz>"
        )
        sys.exit(1)

    vsde_ckpt = Path(sys.argv[1])
    bundle = Path(sys.argv[2])

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    analyze_drift(vsde_ckpt, bundle, device)


if __name__ == "__main__":
    main()
