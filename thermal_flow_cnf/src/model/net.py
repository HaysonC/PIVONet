from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64, depth: int = 3):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(max(0, depth - 1)):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ELU(inplace=True))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
