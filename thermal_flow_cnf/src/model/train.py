from __future__ import annotations

from typing import Iterable

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .base_cnf import CNF
from ..utils.io import save_checkpoint

# --- Physics losses (microfluidics) ---
def no_slip_loss(model: CNF, context: torch.Tensor, H: float, n_samples: int = 128) -> torch.Tensor:
    """Penalize non-zero tangential velocity at channel walls y=±H.

    Approximates by sampling random x, setting y at ±H and evaluating field.
    Loss = mean(|u_y| + |u_x_wall|) where u is predicted field at time t=1.
    (For simple planar channels we expect u≈0 at walls.)
    """
    with torch.no_grad():
        B = context.size(0)
        idx = torch.randint(0, B, (min(n_samples, B),), device=context.device)
    ctx = context[idx]
    x_rand = torch.rand(ctx.size(0), 1, device=ctx.device, dtype=ctx.dtype)
    ys = torch.tensor([[-H], [H]], device=ctx.device, dtype=ctx.dtype).repeat(1, x_rand.size(0)).T  # (K,1)
    # Build two wall batches and concatenate
    z_bottom = torch.cat([x_rand, -torch.ones_like(x_rand) * H], dim=1)
    z_top = torch.cat([x_rand, torch.ones_like(x_rand) * H], dim=1)
    z = torch.cat([z_bottom, z_top], dim=0)
    ctx_full = torch.cat([ctx, ctx], dim=0)
    f = model.eval_field(z, ctx_full, t=1.0)
    # Expect near-zero both components at wall; emphasize normal component (here y) & tangential (x) equally
    return (f.abs()).mean()

def bernoulli_like_loss(model: CNF, context: torch.Tensor, rho: float = 1.0, sample_pairs: int = 64) -> torch.Tensor:
    """Encourage approximate Bernoulli invariance: P + 0.5*rho*|u|^2 ~ const across sampled pairs.

    We don't model pressure P explicitly; approximate by minimizing variance of |u|^2 across batch.
    Loss = var(|u|^2).
    """
    with torch.no_grad():
        B = context.size(0)
        idx = torch.randint(0, B, (min(sample_pairs, B),), device=context.device)
        ctx = context[idx]
        # Sample interior points y ~ Uniform(-0.9H,0.9H) if H present in context (assume second feature maybe y0). Fallback to N(0,1).
    # Heuristic: context may contain (x0, y0, theta). We'll sample around provided y0.
    y0 = ctx[:,1:2]
    x0 = ctx[:,0:1]
    z = torch.cat([x0, y0], dim=1)
    f = model.eval_field(z, ctx, t=1.0)
    speed2 = (f**2).sum(dim=1)
    return speed2.var(unbiased=False)


def compute_log_likelihood(model: CNF, x: torch.Tensor, context: torch.Tensor, steps_jitter: int = 0) -> torch.Tensor:
    # model.log_prob returns (B,1)
    return model.log_prob(x, context, steps=16, stochastic_steps_jitter=steps_jitter)


from torch.utils.data import DataLoader
from typing import Callable, Optional


def train_cnf(
    model: CNF,
    dataloader: DataLoader,  # use DataLoader to ensure Sized
    device: str = "cpu",
    epochs: int = 10,
    lr: float = 1e-3,
    ckpt_dir: str | None = None,
    progress_cb: Optional[Callable[[int, int, int, int, float], None]] = None,
    data_noise_std: float = 0.0,
    steps_jitter: int = 0,
):
    model = model.to(device)
    optim = AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optim, T_max=max(1, epochs))

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        inner = enumerate(dataloader)
        total = len(dataloader)
        for j, batch in inner:
            x_final, _x0, _theta, context = batch
            x_final = x_final.to(device=device, dtype=torch.float32)
            context = context.to(device=device, dtype=torch.float32)

            # Stochastic data augmentation: add small Gaussian jitter to targets
            if data_noise_std and data_noise_std > 0.0:
                x_final = x_final + torch.randn_like(x_final) * float(data_noise_std)

            optim.zero_grad(set_to_none=True)
            log_likelihood = compute_log_likelihood(model, x_final, context, steps_jitter)
            base_loss = -log_likelihood.mean()
            # Physics coefficients optionally passed via context attributes (monkey patch) for now
            phys_loss = torch.tensor(0.0, device=device)
            if hasattr(model, "phys_cfg"):
                cfg = getattr(model, "phys_cfg")
                if cfg.get("no_slip_coef", 0.0) > 0.0 and cfg.get("H", None) is not None:
                    phys_loss = phys_loss + cfg["no_slip_coef"] * no_slip_loss(model, context, H=float(cfg["H"]))
                if cfg.get("bernoulli_coef", 0.0) > 0.0:
                    phys_loss = phys_loss + cfg["bernoulli_coef"] * bernoulli_like_loss(model, context)
            loss = base_loss + phys_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            running += float(loss.item())
            if progress_cb is not None:
                progress_cb(epoch, epochs, j + 1, total, float(loss.item()))

        scheduler.step()
        avg = running / max(1, len(dataloader))
        if ckpt_dir is not None:
            save_checkpoint(model, optim, f"{ckpt_dir}/cnf_epoch{epoch}.pt", epoch=epoch, loss=avg)

    return model
