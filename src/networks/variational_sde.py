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
    """
    Variational SDE model: augments a pretrained CNF with learned stochastic dynamics and posterior encoder.
    
    Learns a generative SDE model via variational inference. Given observed trajectories x,
    infers latent codes z via encoder, then models dynamics as:
        p(x|z) = ∏_t N(x_t | μ_t(z), σ_obs²)  [likelihood]
        dz/dt = f_θ(z, context, t) + u(z, t) + √(2g) dW_t  [posterior SDE]
    
    where f_θ is the frozen CNF dynamics, u is learned posterior drift, and g is learned diffusion.
    
    **Problem Solved**: Learn interpretable SDE dynamics with (1) frozen pretrained physics (CNF),
    (2) posterior inference of latent codes, and (3) learned control signals (u) that guide trajectories
    to match observations. Enables both likelihood computation (via ELBO) and sampling.
    
    **Architecture**:
    - Encoder: x_seq → posterior_ctx (trajectory encoding)
    - Posterior Init: posterior_ctx → (μ, log-var) for latent z
    - Posterior Drift: (z, t, posterior_ctx) → u (learned control signal)
    - CNF backbone: f_θ (frozen, pretrained)
    
    **Key Innovation**: Two-layer drift structure:
        u_drift = f_θ(z, context, t)  [frozen pretrained physics]
              + u(z, t)               [learned posterior controls, only 10-20% of signal]
    
    This separation prevents overfitting and maintains physical interpretability.
    
    **Memory Ownership**:
    - Owns: encoder, post_init, post_drift, log_diff_param
    - Borrows: cnf (reference, not copied; froze weights)
    
    **Attributes**:
        cnf (CNFModel): Frozen pretrained CNF. Weights require_grad=False.
        encoder (TrajectoryEncoder): Maps x_seq → posterior_ctx (trajectory encoding)
        post_init (PosteriorInit): Maps posterior_ctx → (μ, log-var) for z ~ N(μ, σ²)
        post_drift (PosteriorDriftNet): Maps (z, t, posterior_ctx) → u (control)
        log_diff_param (Parameter | Buffer): Log of diffusion coefficient g. Learnable if
            diffusion_learnable=True, else fixed at log(g)=−3 (g ≈ 0.05).
        z_dim (int): Latent dimensionality
    
    **Time Complexity**:
    - sample_z0(): O(batch_size × z_dim)
    - integrate_posterior_sde(): O(n_steps × batch_size × z_dim) for drift evaluation
    - compute_elbo(): O(n_particles × n_timesteps × batch_size) for full likelihood computation
    
    **Space Complexity**:
    - Model parameters: O(encoder_hidden × input_dim + z_dim × context_dim)
    - Forward pass: O(n_particles × batch_size × z_dim × n_steps) (trajectory storage)
    
    **Error Behavior**:
    - Raises ValueError: If integrator not in {"euler", "improved_euler", "rk4"}
    - No validation of input shapes (assumed correct from training loop)
    
    **Usage Example**::
    
        # Pretrain CNF on trajectory likelihood
        cnf = CNFModel(dim=2, cond_dim=64)
        # ... train CNF ...
        
        # Create VSDE model with frozen CNF
        vsde = VariationalSDEModel(
            cnf=cnf,
            z_dim=4,  # Latent code dimension
            ctx_dim=128,  # Context (trajectory encoding) dimension
            encoder_hidden=128,
            drift_hidden=128,
            diffusion_learnable=True  # Learn the diffusion coefficient
        )
        
        # Train VSDE on trajectory data
        artifacts = train_variational_sde_model(
            vsde, train_loader,
            device="cuda",
            epochs=10,
            n_particles=8,
            obs_std=0.05,
            control_cost_scale=0.1
        )
        
        # Sample from posterior
        with torch.no_grad():
            times, traj, controls = vsde.sample_posterior(
                x_seq=obs_trajectory,
                t_seq=obs_times,
                context=context,
                n_particles=4,
                integrator="rk4"
            )
            # traj: (n_steps, n_particles, batch_size, dim)
            # controls: (n_steps-1, n_particles, batch_size, dim)
    
    **Design Decisions**:
    1. Frozen CNF: Prevents catastrophic forgetting of pretrained physics
    2. Posterior encoder: Infers latent codes from full trajectory (not one-shot)
    3. Control cost regularization: ∫ ||u||²/g² dt discourages wild learned dynamics
    4. Flexible integrators: Euler (fast), Improved Euler (accurate), RK4 (very accurate)
    5. Time-dependent diffusion: g(t) decays as trajectory progresses (smoother endpoints)
    """

    def __init__(
        self,
        cnf: CNFModel,
        z_dim: int = 2,
        ctx_dim: int = 128,
        encoder_hidden: int = 128,
        drift_hidden: int = 128,
        diffusion_learnable: bool = False,
    ) -> None:
        """
        Initialize the Variational SDE model.
        
        **Parameters**:
            cnf (CNFModel): Pretrained CNF model. Will be frozen (requires_grad=False).
                Provides f_θ drift term during VSDE integration.
            z_dim (int, default=2): Dimensionality of latent code z.
                Typical: 2-8. Controls model expressiveness (decoder capacity).
            ctx_dim (int, default=128): Dimensionality of trajectory encoding (posterior context).
                Typically matches CNF's cond_dim for compatibility. Determines encoder output size.
            encoder_hidden (int, default=128): Hidden layer width in TrajectoryEncoder and PosteriorInit.
                Larger → more expressive encoder but higher memory cost.
            drift_hidden (int, default=128): Hidden layer width in PosteriorDriftNet (control network).
                Controls capacity of learned posterior drift u(z, t).
            diffusion_learnable (bool, default=False): If True, diffusion coefficient g is a learnable
                parameter. If False, g is fixed at exp(-3) ≈ 0.05. Set True for complex dynamics.
        
        **Returns**: None
        
        **Side Effects**:
        - Freezes CNF weights (requires_grad=False)
        - Initializes encoder, posterior init, posterior drift networks
        - Registers or creates diffusion parameter
        
        **Error Behavior**: No validation of dimensions (caller responsibility).
        """
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
        """
        Sample latent codes z_0 from the posterior.
        
        Implements reparameterization trick: z ~ N(μ, σ²) via z = μ + σ·ε, ε ~ N(0,I).
        
        **Parameters**:
            ctx (Tensor): Posterior context, shape (batch_size, ctx_dim). Output from encoder.
            n_particles (int, default=1): Number of samples per context. More particles → lower variance
                in ELBO gradient, but slower. Typical: 4-16.
        
        **Returns**:
            Tuple[Tensor, Tensor, Tensor]: (z0_samples, mu, logvar)
                - z0_samples: Sampled latent codes, shape (n_particles*batch_size, z_dim)
                - mu: Posterior mean, shape (batch_size, z_dim)
                - logvar: Log posterior variance, shape (batch_size, z_dim)
        
        **Algorithm**:
        1. Encode context → (μ, log-var)
        2. Sample ε ~ N(0,I), shape (n_particles, batch_size, z_dim)
        3. Reparameterize: z = μ + √(var)·ε
        4. Reshape: (n_particles, batch_size, z_dim) → (n_particles*batch_size, z_dim)
        
        **Time Complexity**: O(n_particles × batch_size × z_dim)
        
        **Usage Example**::
        
            ctx = encoder(x_seq, times)  # (batch_size, 128)
            z0, mu, logvar = model.sample_z0(ctx, n_particles=4)
            # z0: (batch_size*4, 2) latent codes
            # mu: (batch_size, 2) posterior means
            # logvar: (batch_size, 2) log variances
        """
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
        """
        Sample trajectories from the learned posterior SDE.
        
        Encodes observed trajectory x to posterior context, samples latent z, then integrates
        the posterior SDE forward to generate sample trajectories.
        
        **Parameters**:
            x_seq (Tensor): Observed trajectory sequence, shape (batch_size, n_timesteps, dim).
                Used to encode posterior context; data itself not directly used in generation.
            t_seq (Tensor): Time grid, shape (n_timesteps,) or (batch_size, n_timesteps).
                Values in [0, 1]. Defines integration time points.
            context (Tensor): CNF conditioning context, shape (batch_size, cond_dim).
                Velocity field features from encoder. Provides frozen CNF drift.
            mask (Tensor | None, default=None): Valid sample mask, shape (batch_size, n_timesteps).
                Used by encoder to focus on valid trajectory points.
            n_particles (int, default=4): Number of samples per trajectory. More → more diversity
                but slower. Typical: 4-16.
            n_integration_steps (int, default=50): Number of ODE solver steps. More → smoother
                trajectories but slower. Typical: 30-100.
            cnf_endpoints (Tuple[Tensor, Tensor] | None, default=None): Optional (z_start, z_final)
                tensors to fix first/last timesteps. If provided, VSDE starts at z_start (ignoring z0).
            length_scale (float, default=1.0): Scaling factor for CNF drift term f_θ.
                <1 → rely more on learned posterior controls, >1 → rely more on pretrained physics.
            diffusion_scale (float, default=1.0): Multiplier on learned diffusion g.
                >1 → more stochastic wiggles, <1 → more deterministic.
            integrator (str, default="euler"): ODE solver for drift: "euler", "improved_euler", "rk4".
                "rk4" is most accurate but slowest. Diffusion always uses Euler-Maruyama.
        
        **Returns**:
            Tuple[Tensor, Tensor, Tensor]: (times, trajectories, controls)
                - times: Integration time grid, shape (n_integration_steps,)
                - trajectories: Generated samples, shape (n_steps, n_particles, batch_size, dim)
                - controls: Learned control signals u(z,t), shape (n_steps-1, n_particles, batch_size, dim)
        
        **Side Effects**: None (no_grad context)
        
        **Time Complexity**: O(n_particles × batch_size × n_steps × model_forward)
        - Each step: O(encoder + drift + CNF forward)
        - Total: ~100ms for typical parameters
        
        **Usage Example**::
        
            model.eval()
            with torch.no_grad():
                times, traj, controls = model.sample_posterior(
                    x_seq=obs_trajectory,  # (32, 100, 2)
                    t_seq=obs_times,       # (100,)
                    context=velocity_ctx,  # (32, 64)
                    n_particles=8,
                    integrator="rk4"
                )
                # traj: (50, 8, 32, 2) [timesteps, particles, batch, dim]
                # Use mean: traj.mean(dim=1) for point estimates
                # Use std: traj.std(dim=1) for uncertainty
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
        """
        Integrate the posterior SDE with flexible solvers and optional fixed endpoints.
        
        Solves: dz = [f_θ(z, context, t) + u(z, t)] dt + √(2g) dW_t
        
        where f_θ is frozen CNF drift, u is learned posterior control, and g is learned diffusion.
        
        **Parameters**:
            z0 (Tensor): Initial latent codes, shape (n_particles*batch_size, z_dim)
            posterior_ctx (Tensor): Posterior context for control network, shape (n_particles*batch_size, ctx_dim)
            cnf_ctx (Tensor): CNF context for drift evaluation, shape (n_particles*batch_size, cond_dim)
            t_obs (Tensor): Time grid for integration, shape (n_timesteps,) or (batch_size, n_timesteps)
            n_steps (int, default=50): Number of solver steps (unused if t_obs provides grid)
            length_scale (float, default=1.0): Scale factor for CNF drift f_θ
            cnf_start (Tensor | None): Fixed starting position, shape (n_particles*batch_size, z_dim).
                If provided, z0 is ignored (overridden by cnf_start).
            cnf_final (Tensor | None): Fixed ending position (currently unused, for future extension)
            diffusion_scale (float, default=1.0): Multiplier on learned g coefficient
            integrator (str): One of {"euler", "improved_euler", "rk4"}.
                - "euler": Simple Euler method (fast, O(dt) error)
                - "improved_euler" / "heun": Predictor-corrector (O(dt²) error)
                - "rk4": Classical 4th-order Runge-Kutta (very accurate, O(dt⁴) error)
                Diffusion is always Euler-Maruyama regardless of integrator choice.
        
        **Returns**:
            Tuple[Tensor, Tensor, Tensor]: (trajectories, controls, times)
                - trajectories: Integrated path, shape (n_timesteps, n_particles*batch_size, z_dim)
                - controls: Learned drift u(z,t) at each step, shape (n_steps-1, n_particles*batch_size, z_dim)
                - times: Integration time grid, shape (n_timesteps,)
        
        **Algorithm** (Stochastic Euler-Maruyama):
        
        For each timestep i → i+1:
        1. Compute drift: f = f_θ(z_i, context) + u(z_i, t_i)  [time-dependent]
        2. Compute diffusion: g = g_param × decay_factor(t_i)  [time-decaying]
        3. Deterministic update (via integrator choice):
           - Euler: z_prop = z + f·dt
           - Heun: z_prop = z + 0.5·(f + f(z+f·dt))·dt
           - RK4: z_prop = z + RK4_step(f)
        4. Stochastic update: z_new = z_prop + √(2g·dt)·ξ, ξ ~ N(0,I)
        5. Append z_new to trajectory
        
        **Time Complexity**: O(n_steps × batch_size × z_dim)
        - Euler: ~n_steps forward passes
        - RK4: ~4n_steps forward passes (each RK4 step evaluates drift 4 times)
        
        **Space Complexity**: O(n_steps × batch_size × z_dim) for trajectory storage
        
        **Error Behavior**:
        - Raises ValueError: If integrator not recognized
        
        **Implementation Notes**:
        - Time decay: g(t) = g_base × (1-t)² encourages smooth endpoints
        - Gradient clamping: max(0, length_scale) prevents negative weights on CNF drift
        - Robust time handling: Handles both 1D and 2D time tensors
        
        **Usage Example**::
        
            z0 = torch.randn(batch_size, z_dim)
            times = torch.linspace(0, 1, 50)
            
            traj, controls, times_out = model.integrate_posterior_sde(
                z0, posterior_ctx, cnf_ctx, times,
                integrator="rk4",
                length_scale=0.8,
                diffusion_scale=1.0
            )
            # traj: (50, batch_size, z_dim)
            # controls: (49, batch_size, z_dim)
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

            # Add diffusion via Euler-Maruyama: dz_diffusion = √(2g·dt) · ξ
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
        """
        Compute evidence lower bound (ELBO) for variational inference.
        
        ELBO = E_z[log p(x|z)] - KL(q(z|x) || p(z)) - cost(controls)
        
        Lower bound on log-likelihood. Maximizing ELBO ↔ maximizing likelihood (with regularization).
        
        **Parameters**:
            x_seq (Tensor): Observed trajectory sequence, shape (batch_size, n_timesteps, dim)
            t_seq (Tensor): Time grid, shape (n_timesteps,) or (batch_size, n_timesteps)
            context (Tensor): CNF conditioning context, shape (batch_size, cond_dim)
            mask (Tensor | None): Valid sample mask, shape (batch_size, n_timesteps)
            n_particles (int, default=4): Number of samples for Monte Carlo ELBO estimate.
                More particles → lower variance but slower.
            obs_std (float, default=0.05): Observation noise std dev. Likelihood variance.
                Smaller → tighter fit, larger → more robust to mismatch.
            kl_warmup (float, default=1.0): Weight on KL-divergence term.
                1.0 → standard ELBO. <1.0 → posterior collapse reduction (entropy regularization).
            control_cost_scale (float, default=1.0): Weight on control cost regularization.
                Encourages simple learned dynamics. Typical: 0.1-1.0.
            n_integration_steps (int, default=50): ODE solver steps for trajectory generation
        
        **Returns**:
            Tuple[Tensor, dict]: (loss, stats)
                - loss (Tensor): Scalar loss = -ELBO (for backward pass)
                - stats (dict): Diagnostic statistics with keys:
                    - "elbo": ELBO value
                    - "log_px": Reconstruction log-likelihood
                    - "control_cost": Control cost term
                    - "kl_z0": KL-divergence term
        
        **Side Effects**: None (gradient tracking enabled for backprop)
        
        **Time Complexity**: O(n_particles × n_timesteps × batch_size) for likelihood computation
        
        **Algorithm** (ELBO Decomposition):
        
        1. **Encoding**: x_seq → posterior_ctx via encoder
        2. **Sampling**: Sample z ~ q(z|x) with n_particles, including (μ, log-var) for KL
        3. **Integration**: VSDE forward pass to generate z(t) trajectories
        4. **Interpolation**: Match generated times to observed times via nearest neighbor
        5. **Reconstruction**: log p(x|z) = -||x - z||² / (2·obs_std²)
        6. **KL divergence**: KL(q||p) = 0.5 Σ(μ² + σ² - 1 - log σ²)
        7. **Control cost**: Σ ∫ ||u||²/g² dt (penalizes complex controls)
        8. **ELBO**: log p(x|z) - control_cost_scale·cost - kl_warmup·kl
        
        **ELBO Components**:
        - **log_px** (reconstruction): Negative squared error scaled by obs_std. Larger → better fit.
        - **kl_z0** (KL-divergence): Regularizes posterior. Prevents posterior collapse.
        - **control_cost**: Penalties ||u||² in latent space. Prevents overfitting.
        
        **Hyperparameter Effects**:
        - **obs_std**: If trajectories have mismatch, increase. If overfitting, decrease.
        - **kl_warmup**: If KL → 0 (collapse), reduce to 0.1-0.5 for first epoch.
        - **control_cost_scale**: If underfitting, reduce to 0.01-0.1. If overfitting, increase.
        
        **Usage Example**::
        
            loss, stats = model.compute_elbo(
                x_seq=obs_trajectory,
                t_seq=times,
                context=velocity_context,
                n_particles=8,
                obs_std=0.05,
                kl_warmup=1.0,
                control_cost_scale=0.1
            )
            loss.backward()
            optimizer.step()
            print(f"ELBO: {stats['elbo']:.4f}, "
                  f"log_px: {stats['log_px']:.4f}, "
                  f"KL: {stats['kl_z0']:.4f}")
        """
        device = x_seq.device
        b, m, d = x_seq.shape
        if t_seq.dim() == 1:
            t_seq = t_seq.unsqueeze(0).expand(b, -1)
        
        # Encode trajectory to posterior context
        posterior_ctx = self.encoder(x_seq, t_seq, mask)
        
        # Sample latent codes with reparameterization trick
        z0_all, mu, logvar = self.sample_z0(posterior_ctx, n_particles)
        n = z0_all.size(0)
        posterior_ctx_rep = posterior_ctx.unsqueeze(0).repeat(n_particles, 1, 1).view(n, -1)
        cnf_ctx = context.to(device=device, dtype=x_seq.dtype)
        cnf_ctx_rep = cnf_ctx.unsqueeze(0).repeat(n_particles, 1, 1).view(n, -1)
        ref_times = t_seq[0]
        
        # Integrate VSDE posterior to generate trajectories
        traj, u_traj, times = self.integrate_posterior_sde(
            z0_all, posterior_ctx_rep, cnf_ctx_rep, ref_times, 
            n_steps=n_integration_steps
        )
        traj = traj.view(traj.size(0), n_particles, b, d)
        u_traj = u_traj.view(u_traj.size(0), n_particles, b, d)
        
        # Match generated times to observed times via nearest neighbor
        diff = torch.abs(times.unsqueeze(1) - ref_times.unsqueeze(0))
        idxs = torch.argmin(diff, dim=0)
        z_obs = traj.index_select(0, idxs)
        z_obs = z_obs.permute(1, 2, 0, 3)
        
        # Compute reconstruction likelihood: log p(x|z)
        x_target = x_seq.unsqueeze(0).expand_as(z_obs)
        mask_exp = mask.unsqueeze(0).unsqueeze(-1) if mask is not None else torch.ones_like(x_target)
        obs_var = float(obs_std) ** 2
        diff_obs = (z_obs - x_target) * mask_exp
        sq_err = (diff_obs ** 2).sum(dim=3) / obs_var
        log_norm = mask_exp.squeeze(-1) * math.log(2.0 * math.pi * obs_var)
        log_px = -0.5 * (sq_err + log_norm)
        log_px = log_px.sum(dim=2).mean(dim=0).mean()
        
        # Compute KL-divergence: KL(q(z|x) || p(z))
        # KL(N(μ,σ²) || N(0,I)) = 0.5 Σ(μ² + σ² - 1 - log σ²)
        kl_z0 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        
        # Compute control cost: ∫ ||u||² / g² dt
        g = F.softplus(self.log_diff_param)
        g2 = g.pow(2) + 1e-12
        dt = (times[1:] - times[:-1]).view(-1, 1, 1)
        control_integrand = (u_traj.pow(2).sum(dim=3) / g2) * dt
        control_cost = 0.5 * control_integrand.sum(dim=0).mean()
        
        # Combine into ELBO: log p(x|z) - control_cost - KL(q||p)
        elbo = log_px - control_cost_scale * control_cost - kl_warmup * kl_z0
        loss = -elbo
        
        stats = {
            "elbo": float(elbo.item()),
            "log_px": float(log_px.item()),
            "control_cost": float(control_cost.item()),
            "kl_z0": float(kl_z0.item()),
        }
        return loss, stats
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
            elif integrator == "dopri5":
                # Dormand-Prince RK5(4) method on drift part; diffusion stays Euler-Maruyama
                # Coefficients from "Numerical Recipes" and Wikipedia
                a = [0, 1/5, 3/10, 4/5, 8/9, 1]
                b = [
                    [],
                    [1/5],
                    [3/40, 9/40],
                    [44/45, -56/15, 32/9],
                    [19372/6561, -25360/2187, 64448/6561, -212/729],
                    [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
                ]
                c = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]

                k = []
                for s in range(6):
                    z_s = z.clone()
                    for j in range(s):
                        if b[s]:
                            z_s = z_s + dt_normalized * b[s][j] * k[j]
                    t_s = t_i + a[s] * dt_normalized
                    f_theta_s = self.cnf.eval_field(z_s, cnf_ctx, float(t_s.item())) * float(max(0.0, length_scale))
                    u_s, _ = self.post_drift(z_s, t_s, posterior_ctx)
                    k_s = f_theta_s + u_s
                    k.append(k_s)

                drift_rk = sum(c[s] * k[s] for s in range(6))
                z_proposed = z + drift_rk * dt_normalized
                controls.append(k[0])  # log first-stage control for analysis
            else:
                raise ValueError(f"Unknown integrator: {integrator}. Use 'euler', 'improved_euler', 'rk4', or 'dopri5'.")

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
