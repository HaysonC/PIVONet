from __future__ import annotations

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

from .net import MLP
from typing import Tuple, cast


class ODEFunc(nn.Module):
    def __init__(self, dim: int, cond_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.net = MLP(dim + cond_dim + 1, dim, hidden_dim)
        self._context: torch.Tensor | None = None

    def set_context(self, context: torch.Tensor):
        # context: (B, cond_dim)
        self._context = context

    def forward(self, t: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # states: (z, logp), each (B, dim) and (B,1)
        z, logp = states
        assert self._context is not None, "Context not set in ODEFunc. Call set_context before integration."
        B = z.size(0)
        tfeat = torch.ones(B, 1, device=z.device, dtype=z.dtype) * t
        inp = torch.cat([z, self._context, tfeat], dim=1)
        f = self.net(inp)
        # Hutchinson trace estimator
        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            eps = torch.randn_like(z)
            f_eps = (self.net(torch.cat([z, self._context.detach(), tfeat.detach()], dim=1)) * eps).sum()
            grad = torch.autograd.grad(f_eps, z, create_graph=True)[0]
            div = (grad * eps).sum(dim=1, keepdim=True)
        dlogp_dt = -div
        return f, dlogp_dt


class CNF(nn.Module):
    def __init__(self, dim: int, cond_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.func = ODEFunc(dim, cond_dim, hidden_dim)

    def flow(self, x: torch.Tensor, context: torch.Tensor, t0: float = 1.0, t1: float = 0.0, steps: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
        # Integrate from data time t0 to base time t1 (usually 1->0)
        B = x.size(0)
        device = x.device
        tspan = torch.linspace(t0, t1, steps=steps, device=device)
        logp0 = torch.zeros(B, 1, device=device, dtype=x.dtype)
        self.func.set_context(context)
        res = odeint(self.func, (x, logp0), tspan, atol=1e-5, rtol=1e-5)
        z_traj, logp_traj = cast(tuple[torch.Tensor, torch.Tensor], res)
        # Each is (T, B, D) and (T, B, 1)
        z1 = z_traj[-1]
        logp1 = logp_traj[-1]
        return z1, logp1

    def log_prob(self, x: torch.Tensor, context: torch.Tensor, base_std: float = 1.0, steps: int = 10):
        # Map x -> z and accumulate logp; base is N(0, I)
        z, delta_logp = self.flow(x, context, t0=1.0, t1=0.0, steps=steps)
        base_logprob = -0.5 * ((z / base_std) ** 2).sum(dim=1, keepdim=True) - 0.5 * self.dim * torch.log(
            torch.tensor(2 * torch.pi, device=x.device, dtype=x.dtype)
        ) - self.dim * torch.log(torch.tensor(base_std, device=x.device, dtype=x.dtype))
        return base_logprob + delta_logp

