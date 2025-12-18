"""Variational trajectory encoder inspired by the old thermal flow code base."""

from __future__ import annotations

import math
from typing import Optional, cast

import torch
import torch.nn as nn

from .mlp import MLP


class FourierTimeEmbedding(nn.Module):
    def __init__(self, emb_dim: int = 32, max_freq: float = 10.0) -> None:
        super().__init__()
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(float(max_freq)), emb_dim // 2)
        )
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.reshape(1, 1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)
        freqs = cast(torch.Tensor, self.freqs)
        args = 2 * math.pi * t * freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TrajectoryEncoder(nn.Module):
    def __init__(
        self, x_dim: int = 2, ctx_dim: int = 64, hidden: int = 128, rnn_layers: int = 1
    ) -> None:
        super().__init__()
        self.time_emb = FourierTimeEmbedding(emb_dim=32)
        self.input_proj = MLP(x_dim + 32, hidden, hidden_dim=hidden, depth=2)
        self.rnn = nn.GRU(
            hidden, hidden, num_layers=rnn_layers, batch_first=True, bidirectional=True
        )
        self.ctx_proj = nn.Linear(2 * hidden, ctx_dim)

    def forward(
        self,
        x_seq: torch.Tensor,
        t_seq: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = x_seq.shape
        t_emb = self.time_emb(t_seq.reshape(B * T, 1)).reshape(B, T, -1)
        inp = torch.cat([x_seq, t_emb], dim=-1)
        h = self.input_proj(inp.reshape(B * T, -1)).reshape(B, T, -1)
        if mask is not None:
            h = h * mask.unsqueeze(-1)
        output, _ = self.rnn(h)
        if mask is not None:
            lengths = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = (output * mask.unsqueeze(-1)).sum(dim=1) / lengths
        else:
            pooled = output.mean(dim=1)
        ctx = torch.tanh(self.ctx_proj(pooled))
        return ctx


class PosteriorInit(nn.Module):
    def __init__(self, ctx_dim: int, z_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = MLP(ctx_dim, 2 * z_dim, hidden_dim=hidden, depth=2)

    def forward(self, ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(ctx)
        mu, logvar = out.chunk(2, dim=-1)
        logvar = torch.clamp(logvar, min=-20.0, max=5.0)
        return mu, logvar


class DiffusionEncoderNet(nn.Module):
    def __init__(self, z_dim: int = 16, ctx_dim: int = 64) -> None:
        super().__init__()
        self.encoder = TrajectoryEncoder(ctx_dim=ctx_dim)
        self.posterior = PosteriorInit(ctx_dim, z_dim)

    def forward(
        self,
        x_seq: torch.Tensor,
        t_seq: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx = self.encoder(x_seq, t_seq, mask)
        mu, logvar = self.posterior(ctx)
        return ctx, mu, logvar
