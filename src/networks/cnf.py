"""Continuous normalizing flow core inspired by the legacy thermal_flow_cnf suite."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

from .mlp import MLP


class ODEFunc(nn.Module):
    def __init__(self, dim: int, cond_dim: int, hidden_dim: int = 64, depth: int = 3, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.net = MLP(dim + cond_dim + 1, dim, hidden_dim=hidden_dim, depth=depth, dropout_p=dropout_p)
        self._context: torch.Tensor | None = None
        self.out_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("z_scale", torch.ones(1, dim))
        self.skip_divergence = False

    def set_context(self, context: torch.Tensor) -> None:
        self._context = context

    def forward(self, t: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._context is None:
            raise RuntimeError("Context must be set before integrating the CNF")
        z, logp = states
        with torch.enable_grad():
            if not z.requires_grad:
                z = z.detach().requires_grad_(True)
            context = self._context.to(device=z.device, dtype=z.dtype)
            t = t.to(device=z.device, dtype=z.dtype)
            t_feat = torch.ones(z.size(0), 1, device=z.device, dtype=z.dtype) * t
            inp = torch.cat([z, context, t_feat], dim=1)
            f = self.net(inp) * self.out_scale
            if self.training and not self.skip_divergence:
                eps = torch.randn_like(z)
                f_eps = (f * eps).sum()
                grad = torch.autograd.grad(f_eps, z, create_graph=True, retain_graph=True, allow_unused=True)[0]
                if grad is None:
                    grad = torch.zeros_like(z)
                div = (grad * eps).sum(dim=1, keepdim=True)
            else:
                div = torch.zeros(z.size(0), 1, device=z.device, dtype=z.dtype)
        return f, -div


class CNFModel(nn.Module):
    """Simplified CNF that learns advection dynamics in the 2D slice."""

    def __init__(
        self,
        dim: int = 2,
        cond_dim: int = 64,
        hidden_dim: int = 128,
        depth: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.func = ODEFunc(dim=dim, cond_dim=cond_dim, hidden_dim=hidden_dim, depth=depth, dropout_p=dropout)

    def _integrate(self, x: torch.Tensor, context: torch.Tensor, tspan: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.to(dtype=torch.float32)
        context = context.to(dtype=torch.float32)
        if not x.requires_grad:
            x = x.requires_grad_()
        logp0 = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
        self.func.set_context(context)
        tspan = tspan.to(device=x.device, dtype=x.dtype)
        # Validate the requested time span to avoid zero-range / underflow errors
        if tspan.numel() < 2:
            raise ValueError("tspan must contain at least two time points for integration")

        t0_val = float(tspan[0].to("cpu"))
        t1_val = float(tspan[-1].to("cpu"))
        if abs(t1_val - t0_val) == 0.0:
            # Nothing to integrate; return initial states unchanged
            return x, logp0

        # Ensure consecutive time differences are non-zero to avoid the
        # ODE solver asserting on an underflow in dt (e.g. "underflow in dt 0.0").
        # This can happen when callers construct a tspan with duplicate time
        # entries or with steps=1. Provide a helpful error message so the
        # orchestrating workflow can be adjusted.
        diffs = torch.abs(tspan[1:] - tspan[:-1])
        if torch.any(diffs <= torch.finfo(tspan.dtype).tiny):
            raise ValueError(
                "Invalid tspan: consecutive time steps contain zero or extremely "
                "small differences which will cause the ODE solver to underflow. "
                "Check caller for t0/t1/steps (received tspan=%s)." % (tspan,)
            )


        solver_options = {"dtype": torch.float32}
        res = odeint(self.func, (x, logp0), tspan, atol=1e-5, rtol=1e-5, options=solver_options)
        assert res is not None, "ODE integration failed during flow"
        z_traj, logp_traj = res
        assert z_traj is not None and logp_traj is not None, "ODE integration returned None states"
        return z_traj[-1], logp_traj[-1]

    def flow(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        *,
        t0: float = 1.0,
        t1: float = 0.0,
        steps: int = 8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tspan = torch.linspace(t0, t1, steps=steps, device=x.device, dtype=x.dtype)
        return self._integrate(x, context, tspan)

    def log_prob(self, x: torch.Tensor, context: torch.Tensor, base_std: float = 1.0) -> torch.Tensor:
        device = x.device
        z, delta_logp = self.flow(x, context)
        normalizer = torch.log(torch.tensor(2 * torch.pi * base_std ** 2, device=device, dtype=z.dtype))
        base = -0.5 * ((z / base_std) ** 2).sum(dim=1, keepdim=True) - 0.5 * self.dim * normalizer
        return (base + delta_logp).squeeze(-1)

    @torch.no_grad()
    def eval_field(self, z: torch.Tensor, context: torch.Tensor, t: float = 1.0) -> torch.Tensor:
        device = z.device
        dtype = z.dtype
        z = z.to(device=device, dtype=dtype)
        context = context.to(device=device, dtype=dtype)
        logp0 = torch.zeros(z.size(0), 1, device=device, dtype=dtype)
        self.func.set_context(context)
        prev = self.func.skip_divergence
        self.func.skip_divergence = True
        try:
            f, _ = self.func(torch.tensor(float(t), device=device, dtype=dtype), (z, logp0))
        finally:
            self.func.skip_divergence = prev
        return f

    @torch.no_grad()
    def sample(self, n: int, context: torch.Tensor, base_std: float = 1.0, steps: int = 30) -> torch.Tensor:
        device = context.device
        dtype = context.dtype
        z0 = base_std * torch.randn(n, self.dim, device=device, dtype=dtype)
        # integrate forward from z0 to x by swapping integration direction
        tspan = torch.linspace(0.0, 1.0, steps=steps, device=device, dtype=dtype)
        self.func.set_context(context)
        assert self.func._context is not None, "Context must be set before sampling"
        solver_options = {"dtype": torch.float32}
        res = odeint(self.func, (z0, torch.zeros(n, 1, device=device, dtype=dtype)), tspan, atol=1e-5, rtol=1e-5, options=solver_options)
        assert res is not None, "ODE integration failed during sampling"
        x_traj, _ = res
        return x_traj[-1]
