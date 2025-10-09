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
        x_final = traj[-1]
        x0 = torch.from_numpy(self.x0s[idx])
        theta = torch.tensor([self.thetas[idx]])
        context = torch.cat([x0, theta], dim=0)  # (3,)
        return x_final, x0, theta, context
