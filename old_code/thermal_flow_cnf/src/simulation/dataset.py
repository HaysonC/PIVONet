from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    def __init__(self, npz_path: str):
        assert os.path.exists(npz_path), f"Dataset not found: {npz_path}"
        data = np.load(npz_path)
        self.x0s = data["x0s"].astype(np.float32)
        self.thetas = data["thetas"].astype(np.float32)
        self.trajs = data["trajs"].astype(np.float32)

    def __len__(self) -> int:
        return self.trajs.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        traj = torch.from_numpy(self.trajs[idx])  # (T+1, 2)
        x_final = traj[-1].to(torch.float32)
        x0 = torch.from_numpy(self.x0s[idx]).to(torch.float32)
        theta = torch.tensor([self.thetas[idx]], dtype=torch.float32)
        context = torch.cat([x0, theta], dim=0).to(torch.float32)  # (3,)
        return x_final, x0, theta, context


class TrajectorySequenceDataset(Dataset):
    """Return entire particle trajectories with their conditioning context and a shared time grid."""

    def __init__(self, npz_path: str, normalize_time: bool = True):
        assert os.path.exists(npz_path), f"Dataset not found: {npz_path}"
        data = np.load(npz_path)
        self.trajs = data["trajs"].astype(np.float32)  # (N, T, 2)
        self.x0s = data["x0s"].astype(np.float32)
        thetas = data["thetas"].astype(np.float32)
        self.contexts = np.concatenate([self.x0s, thetas[:, None]], axis=1).astype(np.float32)
        T = self.trajs.shape[1]
        if normalize_time:
            self.time_grid = torch.linspace(0.0, 1.0, steps=T, dtype=torch.float32)
        else:
            dt = float(data["dt"]) if "dt" in data.files else 1.0
            total = dt * float(max(1, T - 1))
            self.time_grid = torch.linspace(0.0, total, steps=T, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.trajs.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        traj = torch.from_numpy(self.trajs[idx])  # (T, 2)
        context = torch.from_numpy(self.contexts[idx])  # (3,)
        times = self.time_grid.clone()
        mask = torch.ones(traj.size(0), dtype=torch.float32)
        return traj.to(torch.float32), times, context.to(torch.float32), mask
