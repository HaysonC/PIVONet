from __future__ import annotations

import torch


def hutchinson_divergence(fn, x: torch.Tensor):
    """Estimate divergence_x fn(x) using Hutchinson's trick with Rademacher noise.

    Args:
        fn: callable x -> f(x)
        x: (B, D)
    Returns:
        div: (B, 1)
    """
    with torch.set_grad_enabled(True):
        x = x.requires_grad_(True)
        eps = torch.randn_like(x)
        y = (fn(x) * eps).sum()
        grad = torch.autograd.grad(y, x, create_graph=True)[0]
        div = (grad * eps).sum(dim=1, keepdim=True)
    return div
