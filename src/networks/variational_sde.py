"""Variational SDE model that augments a pretrained CNF with diffusion controls."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnf import CNFModel
from .encoder import TrajectoryEncoder, PosteriorInit
from .mlp import MLP


class PosteriorDriftNet(nn.Module):
    def __init__(self, z_dim: int, ctx_dim: int, time_emb_dim: int = 32, hidden: int = 128, out_diff: bool = False) -> None:
        super().__init__()
        self.fourier = FourierTimeEmbedding(time_emb_dim)
        out_dim = z_dim + (z_dim if out_diff else 0)
        self.net = MLP(z_dim + ctx_dim + time_emb_dim, out_dim, hidden_dim=hidden, depth=3)
        self.out_diff = out_diff

    def forward(self, z: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        n = z.size(0)
        if t.dim() == 0:
            t = t.reshape(1, 1).repeat(n, 1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)
        temb = self.fourier(t.view(n, 1))
        inp = torch.cat([z, ctx, temb], dim=-1)
        out = self.net(inp)
        if self.out_diff:
            return out[:, : z.size(1)], out[:, z.size(1) :]
        return out, None


class FourierTimeEmbedding(nn.Module):
    def __init__(self, emb_dim: int = 32, max_freq: float = 10.0) -> None:
        super().__init__()
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(float(max_freq)), emb_dim // 2))
        self.register_buffer("freqs", freqs.view(1, -1))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.reshape(1, 1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)
        args = 2 * math.pi * t * self.freqs # type: ignore
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class VariationalSDEModel(nn.Module):
    def __init__(
        self,
        cnf: CNFModel,
        z_dim: int = 2,
        ctx_dim: int = 128,
        encoder_hidden: int = 128,
        drift_hidden: int = 128,
        diffusion_learnable: bool = False,
    ) -> None:
        super().__init__()
        self.cnf = cnf.eval()  # freeze CNF weights
        for p in self.cnf.parameters():
            p.requires_grad_(False)
        self.encoder = TrajectoryEncoder(ctx_dim=ctx_dim, hidden=encoder_hidden)
        self.post_init = PosteriorInit(ctx_dim, z_dim, hidden=encoder_hidden)
        self.post_drift = PosteriorDriftNet(z_dim, ctx_dim, hidden=drift_hidden)
        if diffusion_learnable:
            self.log_diff_param = nn.Parameter(torch.tensor(-3.0))
        else:
            self.register_buffer("log_diff_param", torch.tensor(-3.0))
        self.z_dim = z_dim

    def sample_z0(self, ctx: torch.Tensor, n_particles: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.post_init(ctx)
        std = torch.exp(0.5 * logvar)
        b, zdim = mu.shape
        eps = torch.randn(n_particles, b, zdim, device=mu.device, dtype=mu.dtype)
        z0 = mu.unsqueeze(0) + eps * std.unsqueeze(0)
        return z0.view(n_particles * b, zdim), mu, logvar

    @torch.no_grad()
    def sample_posterior(
        self,
        x_seq: torch.Tensor,
        t_seq: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        n_particles: int = 4,
        n_integration_steps: int = 50,
        cnf_endpoints: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        length_scale: float = 1.0,
        diffusion_scale: float = 1.0,
        integrator: str = "euler",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample from the posterior with optional fixed CNF endpoints.
        
        Args:
            cnf_endpoints: Optional tuple of (z_start, z_final) from CNF to fix first/last timesteps
            diffusion_scale: Multiplier on learned diffusion to increase/decrease wiggling
        """
        b, t, d = x_seq.shape
        if t_seq.dim() == 1:
            t_seq = t_seq.unsqueeze(0).expand(b, -1)
        posterior_ctx = self.encoder(x_seq, t_seq, mask)
        z0_all, _, _ = self.sample_z0(posterior_ctx, n_particles)
        n = z0_all.size(0)
        posterior_ctx_rep = posterior_ctx.unsqueeze(0).repeat(n_particles, 1, 1).view(n, -1)
        cnf_ctx_rep = context.unsqueeze(0).repeat(n_particles, 1, 1).view(n, -1)
        ref_times = t_seq[0]
        
        # Prepare CNF endpoints if provided
        cnf_start_rep, cnf_final_rep = None, None
        if cnf_endpoints is not None:
            cnf_start, cnf_final = cnf_endpoints
            # Replicate for n_particles: (B, d) -> (n_particles*B, d)
            cnf_start_rep = cnf_start.unsqueeze(0).repeat(n_particles, 1, 1).view(n, -1)
            cnf_final_rep = cnf_final.unsqueeze(0).repeat(n_particles, 1, 1).view(n, -1)
        
        traj, u_traj, times = self.integrate_posterior_sde(
            z0_all, posterior_ctx_rep, cnf_ctx_rep, ref_times, 
            n_steps=n_integration_steps, length_scale=length_scale,
            cnf_start=cnf_start_rep, cnf_final=cnf_final_rep,
            diffusion_scale=diffusion_scale,
            integrator=integrator
        )
        traj = traj.view(traj.size(0), n_particles, b, d)
        u_traj = u_traj.view(u_traj.size(0), n_particles, b, d)
        return times, traj, u_traj

    def integrate_posterior_sde(
        self,
        z0: torch.Tensor,
        posterior_ctx: torch.Tensor,
        cnf_ctx: torch.Tensor,
        t_obs: torch.Tensor,
        n_steps: int = 50,
        length_scale: float = 1.0,
        cnf_start: Optional[torch.Tensor] = None,
        cnf_final: Optional[torch.Tensor] = None,
        diffusion_scale: float = 1.0,
        integrator: str = "euler",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Integrate VSDE with optional fixed CNF endpoints.
        
        Args:
            cnf_start: If provided, fix first timestep to this position (overrides z0)
            cnf_final: If provided, fix last timestep to this position
            diffusion_scale: Multiplier on learned diffusion to increase wiggling
            integrator: One of {"euler", "improved_euler", "rk4"}. For SDEs, diffusion is applied via
                Euler-Maruyama in all modes; the drift component uses the chosen integrator.
        """
        device = z0.device
        dtype = z0.dtype
        # Use the observed time grid if provided to match ground truth resolution.
        # Expect t_obs in normalized [0,1]. If batched, use the first trajectory's times.
        if t_obs.dim() == 2:
            times = t_obs[0].to(device=device, dtype=dtype)
        else:
            times = t_obs.to(device=device, dtype=dtype)
        # If caller passes an explicit n_steps differing from observed, optionally resample.
        # Here we prioritize exact matching to the observation grid.
        if times.numel() < 2:
            # Fallback to a minimal grid
            steps = max(2, int(n_steps))
            times = torch.linspace(0.0, 1.0, steps=steps, device=device, dtype=dtype)
        
        # Initialize first timestep (fixed to CNF start if provided)
        z = cnf_start if cnf_start is not None else z0
        traj = [z]
        controls = []
        g_base = F.softplus(self.log_diff_param) * float(diffusion_scale)  # Base diffusion with scaling
        
        # Integrate all timesteps with VSDE posterior drift + diffusion
        # If CNF endpoints are provided, they constrain where we start and should end up,
        # but VSDE drift naturally guides toward the endpoint (not forced)
        integrator = integrator.lower()
        for i in range(times.numel() - 1):
            t_i = times[i]
            t_next = times[i + 1]
            dt_normalized = t_next - t_i
            
            # Time-dependent diffusion: decay toward endpoint to reduce wild oscillations
            # g(t) = g_base * (1 - t)^2 so diffusion → 0 as t → 1
            time_decay = (1.0 - float(t_i.item())) ** 1
            g = g_base * time_decay

            # Evaluate drift terms
            f_theta = self.cnf.eval_field(z, cnf_ctx, float(t_i.item())) * float(max(0.0, length_scale))
            u, _ = self.post_drift(z, t_i, posterior_ctx)
            drift = f_theta + u

            if integrator == "euler":
                z_proposed = z + drift * dt_normalized
                controls.append(u)
            elif integrator in ("improved_euler", "heun"):
                # Heun's method: predictor-corrector on drift
                z_pred = z + drift * dt_normalized
                f_theta_pred = self.cnf.eval_field(z_pred, cnf_ctx, float(t_next.item())) * float(max(0.0, length_scale))
                u_pred, _ = self.post_drift(z_pred, t_next, posterior_ctx)
                drift_corr = 0.5 * (drift + (f_theta_pred + u_pred))
                z_proposed = z + drift_corr * dt_normalized
                controls.append(u)
            elif integrator == "rk4":
                # Classical RK4 on drift part; diffusion stays Euler-Maruyama
                k1_f = self.cnf.eval_field(z, cnf_ctx, float(t_i.item())) * float(max(0.0, length_scale))
                k1_u, _ = self.post_drift(z, t_i, posterior_ctx)
                k1 = k1_f + k1_u

                z2 = z + 0.5 * dt_normalized * k1
                t2 = t_i + 0.5 * dt_normalized
                k2_f = self.cnf.eval_field(z2, cnf_ctx, float(t2.item())) * float(max(0.0, length_scale))
                k2_u, _ = self.post_drift(z2, t2, posterior_ctx)
                k2 = k2_f + k2_u

                z3 = z + 0.5 * dt_normalized * k2
                t3 = t_i + 0.5 * dt_normalized
                k3_f = self.cnf.eval_field(z3, cnf_ctx, float(t3.item())) * float(max(0.0, length_scale))
                k3_u, _ = self.post_drift(z3, t3, posterior_ctx)
                k3 = k3_f + k3_u

                z4 = z + dt_normalized * k3
                t4 = t_next
                k4_f = self.cnf.eval_field(z4, cnf_ctx, float(t4.item())) * float(max(0.0, length_scale))
                k4_u, _ = self.post_drift(z4, t4, posterior_ctx)
                k4 = k4_f + k4_u

                drift_rk = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
                z_proposed = z + drift_rk * dt_normalized
                controls.append(k1_u)  # log first-stage control for analysis
            else:
                raise ValueError(f"Unknown integrator: {integrator}. Use 'euler', 'improved_euler', or 'rk4'.")

            # Add diffusion via Euler-Maruyama
            xi = torch.randn_like(z)
            noise_scale = torch.sqrt(torch.clamp(dt_normalized, min=1e-12))
            z = z_proposed + xi * (g * noise_scale)
            traj.append(z)
        
        # Don't override the last point - let VSDE integration determine it naturally
        return torch.stack(traj, dim=0), torch.stack(controls, dim=0), times

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
        device = x_seq.device
        b, m, d = x_seq.shape
        if t_seq.dim() == 1:
            t_seq = t_seq.unsqueeze(0).expand(b, -1)
        posterior_ctx = self.encoder(x_seq, t_seq, mask)
        z0_all, mu, logvar = self.sample_z0(posterior_ctx, n_particles)
        n = z0_all.size(0)
        posterior_ctx_rep = posterior_ctx.unsqueeze(0).repeat(n_particles, 1, 1).view(n, -1)
        cnf_ctx = context.to(device=device, dtype=x_seq.dtype)
        cnf_ctx_rep = cnf_ctx.unsqueeze(0).repeat(n_particles, 1, 1).view(n, -1)
        ref_times = t_seq[0]
        traj, u_traj, times = self.integrate_posterior_sde(z0_all, posterior_ctx_rep, cnf_ctx_rep, ref_times, n_steps=n_integration_steps)
        traj = traj.view(traj.size(0), n_particles, b, d)
        u_traj = u_traj.view(u_traj.size(0), n_particles, b, d)
        diff = torch.abs(times.unsqueeze(1) - ref_times.unsqueeze(0))
        idxs = torch.argmin(diff, dim=0)
        z_obs = traj.index_select(0, idxs)
        z_obs = z_obs.permute(1, 2, 0, 3)
        x_target = x_seq.unsqueeze(0).expand_as(z_obs)
        mask_exp = mask.unsqueeze(0).unsqueeze(-1) if mask is not None else torch.ones_like(x_target)
        obs_var = float(obs_std) ** 2
        diff_obs = (z_obs - x_target) * mask_exp
        sq_err = (diff_obs ** 2).sum(dim=3) / obs_var
        log_norm = mask_exp.squeeze(-1) * math.log(2.0 * math.pi * obs_var)
        log_px = -0.5 * (sq_err + log_norm)
        log_px = log_px.sum(dim=2).mean(dim=0).mean()
        kl_z0 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        g = F.softplus(self.log_diff_param)
        g2 = g.pow(2) + 1e-12
        dt = (times[1:] - times[:-1]).view(-1, 1, 1)
        control_integrand = (u_traj.pow(2).sum(dim=3) / g2) * dt
        control_cost = 0.5 * control_integrand.sum(dim=0).mean()
        elbo = log_px - control_cost_scale * control_cost - kl_warmup * kl_z0
        loss = -elbo
        stats = {
            "elbo": float(elbo.item()),
            "log_px": float(log_px.item()),
            "control_cost": float(control_cost.item()),
            "kl_z0": float(kl_z0.item()),
        }
        return loss, stats
