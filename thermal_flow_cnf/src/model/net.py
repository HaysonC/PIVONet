from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64, depth: int = 3, dropout_p: float = 0.0):
        super().__init__()
        layers = []
        d = in_dim
        p = max(0.0, float(dropout_p))
        for _ in range(max(0, depth - 1)):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ELU(inplace=True))
            if p > 0.0:
                layers.append(nn.Dropout(p))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
