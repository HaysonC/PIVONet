from __future__ import annotations

import torch
import torch.nn as nn
from torchdiffeq import odeint as odeint_plain
from torchdiffeq import odeint_adjoint as odeint_adj

from .net import MLP
from typing import Tuple, cast


class ODEFunc(nn.Module):
    def __init__(self, dim: int, cond_dim: int, hidden_dim: int = 64, depth: int = 3, dropout_p: float = 0.0, trace_samples: int = 1):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.net = MLP(dim + cond_dim + 1, dim, hidden_dim, depth=depth, dropout_p=dropout_p)
        self._context: torch.Tensor | None = None
        self.trace_samples = int(max(1, trace_samples))
        # Normalization buffers (z and context)
        self.register_buffer("z_mu", torch.zeros(1, dim))
        self.register_buffer("z_sigma", torch.ones(1, dim))
        self.register_buffer("ctx_mu", torch.zeros(1, cond_dim))
        self.register_buffer("ctx_sigma", torch.ones(1, cond_dim))
        # Global output scale (learnable gain)
        self.out_scale = nn.Parameter(torch.tensor(1.0))
        # Flag to skip divergence; simple Python attribute (avoid type annotation for lint)
        self.skip_divergence = False

    def set_context(self, context: torch.Tensor):
        # context: (B, cond_dim)
        self._context = context

    def forward(self, t: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # states: (z, logp), each (B, dim) and (B,1)
        z, logp = states
        assert self._context is not None, "Context not set in ODEFunc. Call set_context before integration."
        B = z.size(0)
        # Ensure time and context are same dtype/device as z to avoid upcasting on MPS
        t = t.to(device=z.device, dtype=z.dtype)
        tfeat = torch.ones(B, 1, device=z.device, dtype=z.dtype) * t
        context = self._context.to(device=z.device, dtype=z.dtype)
        # (Optional) normalization currently disabled for stability; treat as identity
        z_n = z
        ctx_n = context
        # Fast path for inference/no_grad to avoid autograd graph creation
        if not torch.is_grad_enabled():
            f = self.net(torch.cat([z_n, ctx_n, tfeat], dim=1)) * self.out_scale
            dlogp_dt = torch.zeros(B, 1, device=z.device, dtype=z.dtype)
            return f, dlogp_dt

        # Training / grad-enabled path
        z_req = z.requires_grad_(True)
        f = self.net(torch.cat([z_req, ctx_n, tfeat], dim=1)) * self.out_scale
        if getattr(self, "skip_divergence", False):
            dlogp_dt = torch.zeros(B, 1, device=z.device, dtype=z.dtype)
            return f, dlogp_dt
        div_terms: list[torch.Tensor] = []
        for k in range(self.trace_samples):
            eps = torch.randn_like(z_req)
            f_eps = (f * eps).sum()
            grad = torch.autograd.grad(
                f_eps,
                z_req,
                create_graph=True,
                retain_graph=True,
            )[0]
            div_terms.append((grad * eps).sum(dim=1, keepdim=True))
        div = torch.stack(div_terms, dim=0).mean(dim=0)
        dlogp_dt = -div
        return f, dlogp_dt


class CNF(nn.Module):
    def eval_field(self, z: torch.Tensor, context: torch.Tensor, t: float = 1.0) -> torch.Tensor:
        """Evaluate learned vector field f(t, z | context) at a scalar time t.

        Args:
            z: (B, dim)
            context: (B, cond_dim)
            t: scalar time in [0,1]
        Returns:
            f: (B, dim)
        """
        device = z.device
        dtype = z.dtype if z.dtype in (torch.float32, torch.float16, torch.bfloat16) else torch.float32
        z = z.to(device=device, dtype=dtype)
        context = context.to(device=device, dtype=dtype)
        self.func.set_context(context)
        logp0 = torch.zeros(z.size(0), 1, device=device, dtype=dtype)
        # Skip divergence inside forward for eval-only field queries
        prev = getattr(self.func, "skip_divergence", False)
        setattr(self.func, "skip_divergence", True)
        try:
            f, _ = self.func(torch.tensor(float(t), device=device, dtype=dtype), (z, logp0))
        finally:
            setattr(self.func, "skip_divergence", prev)
        return f
    def _integrate_fixed(self, x: torch.Tensor, logp: torch.Tensor, tspan: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Fixed-step RK4 integrator compatible with autograd
        assert tspan.ndim == 1 and tspan.numel() >= 2
        z = x
        lp = logp
        traj_z = [z]
        traj_lp = [lp]
        for i in range(tspan.numel() - 1):
            ti = tspan[i]
            tf = tspan[i + 1]
            h = tf - ti
            k1_z, k1_lp = self.func(ti, (z, lp))
            k2_z, k2_lp = self.func(ti + 0.5 * h, (z + 0.5 * h * k1_z, lp + 0.5 * h * k1_lp))
            k3_z, k3_lp = self.func(ti + 0.5 * h, (z + 0.5 * h * k2_z, lp + 0.5 * h * k2_lp))
            k4_z, k4_lp = self.func(tf, (z + h * k3_z, lp + h * k3_lp))
            z = z + (h / 6.0) * (k1_z + 2.0 * k2_z + 2.0 * k3_z + k4_z)
            lp = lp + (h / 6.0) * (k1_lp + 2.0 * k2_lp + 2.0 * k3_lp + k4_lp)
            traj_z.append(z)
            traj_lp.append(lp)
        return torch.stack(traj_z, dim=0), torch.stack(traj_lp, dim=0)
    def __init__(self, dim: int, cond_dim: int, hidden_dim: int = 64, depth: int = 3, dropout_p: float = 0.0, trace_samples: int = 1):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.func = ODEFunc(dim, cond_dim, hidden_dim, depth=depth, dropout_p=dropout_p, trace_samples=trace_samples)

    def set_normalization(self, z_mu: torch.Tensor, z_sigma: torch.Tensor, ctx_mu: torch.Tensor, ctx_sigma: torch.Tensor):
        """(Disabled) Previously configured normalization stats. Currently a no-op to avoid buffer typing issues."""
        return

    def set_output_scale(self, scale: float):
        with torch.no_grad():
            self.func.out_scale.fill_(float(scale))

    def flow(self, x: torch.Tensor, context: torch.Tensor, t0: float = 1.0, t1: float = 0.0, steps: int = 10, stochastic_steps_jitter: int = 0, solver: str = "dopri5", rtol: float = 1e-5, atol: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
        # Integrate from data time t0 to base time t1 (usually 1->0)
        B = x.size(0)
        device = x.device
        dtype = x.dtype if x.dtype in (torch.float32, torch.float16, torch.bfloat16) else torch.float32
        x = x.to(device=device, dtype=dtype)
        context = context.to(device=device, dtype=dtype)
        # Optionally randomize number of steps per call for stochastic training
        if stochastic_steps_jitter and self.training:
            s = int(max(2, steps + torch.randint(low=-stochastic_steps_jitter, high=stochastic_steps_jitter + 1, size=(1,)).item()))
        else:
            s = int(steps)
        tspan = torch.linspace(t0, t1, steps=s, device=device, dtype=dtype)
        logp0 = torch.zeros(B, 1, device=device, dtype=dtype)
        self.func.set_context(context)
        if x.device.type == "mps":
            # Use fixed-step RK4 on MPS to avoid dtype issues inside torchdiffeq
            res = self._integrate_fixed(x, logp0, tspan)
        else:
            res = odeint_adj(self.func, (x, logp0), tspan, atol=atol, rtol=rtol, method=solver)
        z_traj, logp_traj = cast(tuple[torch.Tensor, torch.Tensor], res)
        # Each is (T, B, D) and (T, B, 1)
        z1 = z_traj[-1]
        logp1 = logp_traj[-1]
        return z1, logp1

    def log_prob(self, x: torch.Tensor, context: torch.Tensor, base_std: float = 1.0, steps: int = 10, stochastic_steps_jitter: int = 0, solver: str = "dopri5", rtol: float = 1e-5, atol: float = 1e-5):
        # Map x -> z and accumulate logp; base is N(0, I)
        z, delta_logp = self.flow(x, context, t0=1.0, t1=0.0, steps=steps, stochastic_steps_jitter=stochastic_steps_jitter, solver=solver, rtol=rtol, atol=atol)
        base_logprob = -0.5 * ((z / base_std) ** 2).sum(dim=1, keepdim=True) - 0.5 * self.dim * torch.log(
            torch.tensor(2 * torch.pi, device=x.device, dtype=x.dtype)
        ) - self.dim * torch.log(torch.tensor(base_std, device=x.device, dtype=x.dtype))
        return base_logprob + delta_logp

    @torch.no_grad()
    def sample(self, n: int, context: torch.Tensor, base_std: float = 1.0, steps: int = 10, H: float | None = None, enforce_bounds: bool = True) -> torch.Tensor:
        device = context.device
        dtype = context.dtype if context.dtype in (torch.float32, torch.float16, torch.bfloat16) else torch.float32
        context = context.to(device=device, dtype=dtype)
        z0 = base_std * torch.randn(n, self.dim, device=device, dtype=dtype)
        # integrate forward from base time 0 -> 1 by flipping signs in ODE: reuse flow with t0=0,t1=1 by negating vector field via small wrapper
        # Simple approach: integrate reverse by negating dlogp but same direction for z; for visualization we only need z path
        # Here we reuse flow with swapped times by using negative net: approximate by running flow on -z and then negating back
        # For correctness, a separate reverse ODEFunc is ideal; to keep scope focused, we approximate by solving with t0=0,t1=1 using current func.
        # Use odeint directly with current func and set initial state at base.
        self.func.set_context(context)
        tspan = torch.linspace(0.0, 1.0, steps=steps, device=device, dtype=dtype)
        logp0 = torch.zeros(n, 1, device=device, dtype=dtype)
        if context.device.type == "mps":
            z_traj, _ = self._integrate_fixed(z0, logp0, tspan)
        else:
            z_traj, _ = cast(tuple[torch.Tensor, torch.Tensor], odeint_adj(self.func, (z0, logp0), tspan, atol=1e-5, rtol=1e-5))
        x = z_traj[-1]
        # Enforce simple reflecting boundary in y to keep samples within [-H, H]
        if enforce_bounds and H is not None:
            y = x[:, 1]
            y = torch.where(y > H, 2 * H - y, y)
            y = torch.where(y < -H, -2 * H - y, y)
            x = torch.stack([x[:, 0], y], dim=1)
        return x

    @torch.no_grad()
    def sample_trajectories(self, n: int, context: torch.Tensor, base_std: float = 1.0, steps: int = 50, H: float | None = None, enforce_bounds: bool = True) -> torch.Tensor:
        device = context.device
        dtype = context.dtype if context.dtype in (torch.float32, torch.float16, torch.bfloat16) else torch.float32
        context = context.to(device=device, dtype=dtype)
        z0 = base_std * torch.randn(n, self.dim, device=device, dtype=dtype)
        self.func.set_context(context)
        tspan = torch.linspace(0.0, 1.0, steps=steps, device=device, dtype=dtype)
        logp0 = torch.zeros(n, 1, device=device, dtype=dtype)
        if context.device.type == "mps":
            z_traj, _ = self._integrate_fixed(z0, logp0, tspan)
        else:
            z_traj, _ = cast(tuple[torch.Tensor, torch.Tensor], odeint_adj(self.func, (z0, logp0), tspan, atol=1e-5, rtol=1e-5))
        # z_traj: (T, B, D) -> return (B, T, D)
        out = z_traj.permute(1, 0, 2).contiguous()
        if enforce_bounds and H is not None:
            y = out[:, :, 1]
            y = torch.where(y > H, 2 * H - y, y)
            y = torch.where(y < -H, -2 * H - y, y)
            out = torch.stack([out[:, :, 0], y], dim=2)
        return out

    @torch.no_grad()
    def rollout_from(self, x0: torch.Tensor, context: torch.Tensor, steps: int = 50, H: float | None = None, enforce_bounds: bool = True) -> torch.Tensor:
        """Generate trajectories starting from given initial positions x0 following learned field.

        Args:
            x0: (B, dim) initial positions in data space
            context: (B, cond_dim)
            steps: number of integration steps between t in [0,1]
            H: optional half-height to reflect y within [-H,H]
            enforce_bounds: if True, apply simple reflecting bounds in y
        Returns:
            (B, steps, dim) trajectory including the initial state as step 0
        """
        device = x0.device
        dtype = x0.dtype if x0.dtype in (torch.float32, torch.float16, torch.bfloat16) else torch.float32
        x0 = x0.to(device=device, dtype=dtype)
        context = context.to(device=device, dtype=dtype)
        self.func.set_context(context)
        tspan = torch.linspace(0.0, 1.0, steps=max(2, int(steps)), device=device, dtype=dtype)
        logp0 = torch.zeros(x0.size(0), 1, device=device, dtype=dtype)
        if device.type == "mps":
            z_traj, _ = self._integrate_fixed(x0, logp0, tspan)
        else:
            z_traj, _ = cast(tuple[torch.Tensor, torch.Tensor], odeint_adj(self.func, (x0, logp0), tspan, atol=1e-5, rtol=1e-5))
        out = z_traj.permute(1, 0, 2).contiguous()
        if enforce_bounds and H is not None:
            y = out[:, :, 1]
            y = torch.where(y > H, 2 * H - y, y)
            y = torch.where(y < -H, -2 * H - y, y)
            out = torch.stack([out[:, :, 0], y], dim=2)
        return out

