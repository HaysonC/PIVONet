"""Helper MLP layers shared across CNF and encoder modules."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 64,
        depth: int = 3,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            *self._build_layers(in_dim, out_dim, hidden_dim, depth, dropout_p)
        )

    def _build_layers(
        self, in_dim: int, out_dim: int, hidden_dim: int, depth: int, dropout_p: float
    ) -> Sequence[nn.Module]:
        modules: list[nn.Module] = []
        dims = [in_dim] + [hidden_dim] * max(depth - 1, 0) + [out_dim]
        for src, dst in zip(dims[:-1], dims[1:]):
            modules.append(nn.Linear(src, dst))
            if dst != out_dim:
                modules.append(nn.ReLU(inplace=True))
                if dropout_p > 0.0:
                    modules.append(nn.Dropout(dropout_p))
        return modules

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
