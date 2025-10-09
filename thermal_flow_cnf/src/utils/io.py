from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np
import torch


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: str, **extra):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
    }
    payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer | None, path: str):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])
    return ckpt


def save_npz(path: str, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_npz(path: str) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}
