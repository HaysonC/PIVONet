from __future__ import annotations

import math
from typing import Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from .net import MLP

if TYPE_CHECKING:
    from .base_cnf import CNF


class FourierTimeEmbedding(nn.Module):
    """Sinusoidal / Fourier time embedding for scalar times.

    Maps t (B,1) or scalar to embedding (B, emb_dim).
    """

    def __init__(self, emb_dim: int = 32, max_freq: float = 10.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.max_freq = float(max_freq)
        # frequencies loglinearly spaced
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(self.max_freq), emb_dim // 2))
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,1) or (B,) or scalar
        if t.dim() == 0:
            t = t.reshape(1, 1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)
        # (B, 1) * (F,) -> (B, F)
        t = t.to(dtype=self.freqs.dtype)  # type: ignore[arg-type]
        args = 2.0 * math.pi * t * self.freqs  # type: ignore[operator]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb


class BiGRUEncoder(nn.Module):
    """Temporal encoder for irregularly-sampled trajectories.

    Inputs:
      x_seq: (B, M, x_dim)
      t_seq: (B, M) timestamps in [0,1]
      mask: (B, M) optional mask (1=valid)

    Outputs:
      context: (B, ctx_dim)
    """

    def __init__(self, x_dim: int = 2, time_emb_dim: int = 32, rnn_hidden: int = 128, rnn_layers: int = 1, ctx_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.time_emb = FourierTimeEmbedding(time_emb_dim)
        self.input_mlp = MLP(x_dim + time_emb_dim, rnn_hidden, hidden_dim=rnn_hidden, depth=2)
        self.rnn = nn.GRU(rnn_hidden, rnn_hidden, num_layers=rnn_layers, batch_first=True, bidirectional=True, dropout=dropout if rnn_layers > 1 else 0.0)
        self.ctx_proj = nn.Linear(2 * rnn_hidden, ctx_dim)

    def forward(self, x_seq: torch.Tensor, t_seq: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x_seq: (B, M, x_dim)
        B, M, _ = x_seq.shape
        t_emb = self.time_emb(t_seq.view(B * M, 1)).view(B, M, -1)
        inp = torch.cat([x_seq, t_emb], dim=-1)
        h = self.input_mlp(inp.view(B * M, -1)).view(B, M, -1)
        if mask is not None:
            # pack padded sequence would be better, but require lengths; simple masking: zero-out invalid
            h = h * mask.unsqueeze(-1)
        out, hn = self.rnn(h)
        # take last timestep outputs combined (bidirectional -> concat of forward/backward)
        # We pool by mean over valid time steps as robust summary
        if mask is not None:
            lengths = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = (out * mask.unsqueeze(-1)).sum(dim=1) / lengths
        else:
            pooled = out.mean(dim=1)
        ctx = torch.tanh(self.ctx_proj(pooled))
        return ctx


class PosteriorInit(nn.Module):
    """Predict initial latent mean and log-variance from context."""

    def __init__(self, ctx_dim: int, z_dim: int, hidden: int = 128):
        super().__init__()
        self.net = MLP(ctx_dim, 2 * z_dim, hidden_dim=hidden, depth=2)

    def forward(self, ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(ctx)
        mu, logvar = out.chunk(2, dim=-1)
        # clamp logvar for numerical stability
        logvar = torch.clamp(logvar, min=-20.0, max=5.0)
        return mu, logvar


class PosteriorDriftNet(nn.Module):
    """u_phi(z, t, context) optionally outputs drift correction and log diffusion adjustment."""

    def __init__(self, z_dim: int, ctx_dim: int, time_emb_dim: int = 32, hidden: int = 128, out_diff: bool = False):
        super().__init__()
        self.time_emb = FourierTimeEmbedding(time_emb_dim)
        self.net = MLP(z_dim + ctx_dim + time_emb_dim, z_dim + (z_dim if out_diff else 0), hidden_dim=hidden, depth=3)
        self.out_diff = out_diff

    def forward(self, z: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # z: (N, z_dim), t: scalar or (N,1), ctx: (N, ctx_dim)
        N = z.size(0)
        if t.dim() == 0:
            t = t.reshape(1, 1).repeat(N, 1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)
        temb = self.time_emb(t.view(N, 1))
        inp = torch.cat([z, ctx, temb], dim=-1)
        out = self.net(inp)
        if self.out_diff:
            u = out[:, : z.size(1)]
            log_s = out[:, z.size(1) :]
            return u, log_s
        else:
            return out, None


class VariationalSDEModel(nn.Module):
    """Top-level module combining encoder, posterior networks and SDE integrator.

    Parameters:
        cnf: pretrained CNF model (θ) used for base drift f_theta and decoder.
        z_dim: latent dimension (should match CNF dim if decoding via CNF reverse mapping).
    """

    def __init__(
        self,
        cnf: CNF,
        z_dim: int,
        ctx_dim: int = 128,
        encoder_cfg: Optional[dict] = None,
        init_hidden: int = 128,
        drift_hidden: int = 128,
        diffusion_learnable: bool = False,
    ):
        super().__init__()
        enc_cfg = encoder_cfg or {}
        self.encoder = BiGRUEncoder(x_dim=enc_cfg.get("x_dim", 2), time_emb_dim=enc_cfg.get("time_emb_dim", 32), rnn_hidden=enc_cfg.get("rnn_hidden", 128), rnn_layers=enc_cfg.get("rnn_layers", 1), ctx_dim=ctx_dim)
        self.post_init = PosteriorInit(ctx_dim, z_dim, hidden=init_hidden)
        self.post_drift = PosteriorDriftNet(z_dim, ctx_dim, time_emb_dim=enc_cfg.get("time_emb_dim", 32), hidden=drift_hidden, out_diff=False)
        # if diffusion learnable, a positive scalar per-dim or global
        if diffusion_learnable:
            self.log_diff_param = nn.Parameter(torch.tensor(-3.0))
        else:
            self.register_buffer("log_diff_param", torch.tensor(-3.0))
        self.cnf = cnf
        self.z_dim = int(z_dim)
        self.cond_dim = getattr(cnf, "cond_dim", None)
        # Freeze CNF parameters – we treat it as a fixed dynamics model during variational training.
        for p in self.cnf.parameters():
            p.requires_grad_(False)
        self.cnf.eval()

    def sample_z0(self, ctx: torch.Tensor, n_particles: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # returns z0: (P*B, z_dim), mu:(B,z), logvar:(B,z)
        mu, logvar = self.post_init(ctx)
        std = torch.exp(0.5 * logvar)
        B, zdim = mu.shape
        eps = torch.randn(n_particles, B, zdim, device=mu.device, dtype=mu.dtype)
        z0 = mu.unsqueeze(0) + eps * std.unsqueeze(0)
        # reshape to (P*B, zdim) for parallel integration
        z0 = z0.view(n_particles * B, zdim)
        return z0, mu, logvar

    @torch.no_grad()
    def sample_posterior(
        self,
        x_seq: torch.Tensor,
        t_seq: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        n_particles: int = 4,
        n_integration_steps: int = 50,
        align_to_observations: bool = True,
        anchor_noise: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return posterior SDE rollouts conditioned on an observed trajectory batch.

        Returns:
            times: (T,)
            traj: (T, P, B, D)
            controls: (T-1, P, B, D)
        """

        device = x_seq.device
        B, M, D = x_seq.shape
        if t_seq.dim() == 1:
            t_seq = t_seq.unsqueeze(0).expand(B, -1)
        t_seq = t_seq.to(device=device, dtype=x_seq.dtype)
        if mask is not None:
            mask = mask.to(device=device, dtype=x_seq.dtype)

        posterior_ctx = self.encoder(x_seq, t_seq, mask)
        if align_to_observations:
            x0 = x_seq[:, 0, :]  # (B, D)
            if anchor_noise > 0.0:
                noise = anchor_noise * torch.randn(n_particles, B, x0.size(1), device=device, dtype=x_seq.dtype)
                z0 = x0.unsqueeze(0) + noise
            else:
                z0 = x0.unsqueeze(0).expand(n_particles, -1, -1)
            z0_all = z0.reshape(n_particles * B, -1)
        else:
            z0_all, _, _ = self.sample_z0(posterior_ctx, n_particles)
        N = z0_all.size(0)
        posterior_ctx_rep = posterior_ctx.unsqueeze(0).repeat(n_particles, 1, 1).view(N, -1)
        cnf_ctx = context.to(device=device, dtype=x_seq.dtype)
        cnf_ctx_rep = cnf_ctx.unsqueeze(0).repeat(n_particles, 1, 1).view(N, -1)

        ref_times = t_seq[0]
        traj, u_traj, times = self.integrate_posterior_sde(
            z0_all,
            posterior_ctx_rep,
            cnf_ctx_rep,
            ref_times,
            n_steps=n_integration_steps,
        )
        T = traj.size(0)
        traj = traj.view(T, n_particles, B, D)
        u_traj = u_traj.view(T - 1, n_particles, B, D)
        return times, traj, u_traj

    def integrate_posterior_sde(
        self,
        z0: torch.Tensor,
        posterior_ctx: torch.Tensor,
        cnf_ctx: torch.Tensor,
        t_obs: torch.Tensor,
        n_steps: int = 50,
        scheme: str = "euler",
        dt_min: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Integrate posterior SDE forward and return:
           z_traj: (T, P*B, z), u_traj: (T, P*B, z), times: (T,)

        t_obs: tensor of observation times in [0,1], shape (M,)
        """
        device = z0.device
        dtype = z0.dtype
        t_obs = t_obs.to(device=device, dtype=dtype) if t_obs.numel() > 0 else t_obs
        steps = max(2, int(n_steps))
        times = torch.linspace(0.0, 1.0, steps=steps, device=device, dtype=dtype)
        T = times.numel()
        z = z0
        traj = [z]
        u_traj = []
        # diffusion magnitude (per-dim scalar)
        g = F.softplus(self.log_diff_param)
        for i in range(T - 1):
            t_i = times[i]
            dt = times[i + 1] - t_i
            if dt_min is not None:
                dt = torch.clamp(dt, min=float(dt_min))
            # evaluate base drift f_theta via cnf.eval_field; expect cnf_ctx shape (P*B, cond_dim)
            f_theta = self.cnf.eval_field(z, cnf_ctx, float(t_i.item()))
            u, _ = self.post_drift(z, t_i, posterior_ctx)
            drift = f_theta + u
            # Euler-Maruyama
            if scheme == "euler":
                xi = torch.randn_like(z)
                noise_scale = torch.sqrt(torch.clamp(dt, min=1e-12))
                z = z + drift * dt + xi * (g * noise_scale)
            else:
                # simple Heun predictor-corrector for SDE (explicit only)
                xi = torch.randn_like(z)
                noise_scale = torch.sqrt(torch.clamp(dt, min=1e-12))
                z_pred = z + drift * dt + xi * (g * noise_scale)
                f_theta_pred = self.cnf.eval_field(z_pred, cnf_ctx, float((t_i + dt).item()))
                u_pred, _ = self.post_drift(z_pred, t_i + dt, posterior_ctx)
                drift_pred = f_theta_pred + u_pred
                z = z + 0.5 * (drift + drift_pred) * dt + xi * (g * noise_scale)
            traj.append(z)
            u_traj.append(u)

        traj = torch.stack(traj, dim=0)  # (T, N, z)
        u_traj = torch.stack(u_traj, dim=0)  # (T-1, N, z)
        return traj, u_traj, times

    def compute_elbo(
        self,
        x_seq: torch.Tensor,
        t_seq: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        n_particles: int = 4,
        obs_std: float = 0.05,
        kl_warmup: float = 1.0,
        control_cost_scale: float = 1.0,
        n_integration_steps: int = 50,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute Monte-Carlo ELBO estimate for a batch.

        Returns loss (to minimize) and a dict with diagnostics.
        """
        device = x_seq.device
        B, M, D = x_seq.shape
        if t_seq.dim() == 1:
            t_seq = t_seq.unsqueeze(0).expand(B, -1)
        t_seq = t_seq.to(device=device, dtype=x_seq.dtype)
        if mask is not None:
            mask = mask.to(device=device, dtype=x_seq.dtype)
        # Ensure all samples share the same observation grid (current implementation assumption)
        ref_times = t_seq[0]
        if not torch.allclose(t_seq, ref_times.unsqueeze(0).expand_as(t_seq)):
            raise ValueError("VariationalSDEModel.compute_elbo expects a shared time grid across the batch.")

        # Encode amortised posterior context
        posterior_ctx = self.encoder(x_seq, t_seq, mask)
        z0_all, mu, logvar = self.sample_z0(posterior_ctx, n_particles)
        N = z0_all.size(0)
        # Repeat contexts for particles
        posterior_ctx_rep = posterior_ctx.unsqueeze(0).repeat(n_particles, 1, 1).view(N, -1)
        cnf_ctx = context.to(device=device, dtype=x_seq.dtype)
        cnf_ctx_rep = cnf_ctx.unsqueeze(0).repeat(n_particles, 1, 1).view(N, -1)

        traj, u_traj, times = self.integrate_posterior_sde(
            z0_all,
            posterior_ctx_rep,
            cnf_ctx_rep,
            ref_times,
            n_steps=n_integration_steps,
        )

        # Reshape trajectories for convenience
        T = traj.size(0)
        traj = traj.view(T, n_particles, B, D)
        u_traj = u_traj.view(T - 1, n_particles, B, D)

        # Align SDE trajectory with observation times via nearest neighbour on the integration grid
        diff = torch.abs(times.unsqueeze(1) - ref_times.unsqueeze(0))
        idxs = torch.argmin(diff, dim=0)  # (M,)
        z_obs = traj.index_select(0, idxs)  # (M, P, B, D)
        z_obs = z_obs.permute(1, 2, 0, 3)  # (P, B, M, D)

        x_target = x_seq.unsqueeze(0).expand(n_particles, -1, -1, -1)
        if mask is not None:
            mask_exp = mask.unsqueeze(0).unsqueeze(-1)
        else:
            mask_exp = torch.ones(1, B, M, 1, device=device, dtype=x_seq.dtype)

        obs_var = float(obs_std) ** 2
        diff_obs = (z_obs - x_target) * mask_exp
        sq_err = (diff_obs ** 2).sum(dim=3) / obs_var
        log_norm = mask_exp.squeeze(-1) * math.log(2.0 * math.pi * obs_var)
        log_px = -0.5 * (sq_err + log_norm)
        log_px = log_px.sum(dim=2)  # (P, B)
        expected_log_px = log_px.mean(dim=0).mean()

        # KL divergence between amortised posterior q(z0|x) and standard normal prior p(z0)
        kl_z0 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

        # Control (energy) cost for drift adjustments
        g = F.softplus(self.log_diff_param)
        g2 = g.pow(2) + 1e-12
        dt = (times[1:] - times[:-1]).view(-1, 1, 1)
        control_integrand = (u_traj.pow(2).sum(dim=3) / g2) * dt
        control_cost = 0.5 * control_integrand.sum(dim=0).mean()

        elbo = expected_log_px - control_cost_scale * control_cost - kl_warmup * kl_z0
        loss = -elbo
        stats = {
            "elbo": float(elbo.item()),
            "log_px_mean": float(log_px.mean().item()),
            "control_cost": float(control_cost.item()),
            "kl_z0": float(kl_z0.item()),
            "z_traj_std": float(traj.std().item()),
        }
        return loss, stats


__all__ = ["VariationalSDEModel", "BiGRUEncoder", "PosteriorInit", "PosteriorDriftNet"]
