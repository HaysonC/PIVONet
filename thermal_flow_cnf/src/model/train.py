from __future__ import annotations

from typing import Iterable

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .base_cnf import CNF
from ..utils.io import save_checkpoint


def compute_log_likelihood(model: CNF, x: torch.Tensor, context: torch.Tensor, steps_jitter: int = 0) -> torch.Tensor:
    # model.log_prob returns (B,1)
    return model.log_prob(x, context, steps=16, stochastic_steps_jitter=steps_jitter)


from torch.utils.data import DataLoader
from typing import Callable, Optional


def train_cnf(
    model: CNF,
    dataloader: DataLoader,  # use DataLoader to ensure Sized
    device: str = "cpu",
    epochs: int = 10,
    lr: float = 1e-3,
    ckpt_dir: str | None = None,
    progress_cb: Optional[Callable[[int, int, int, int, float], None]] = None,
    data_noise_std: float = 0.0,
    steps_jitter: int = 0,
):
    model = model.to(device)
    optim = AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optim, T_max=max(1, epochs))

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        inner = enumerate(dataloader)
        total = len(dataloader)
        for j, batch in inner:
            x_final, _x0, _theta, context = batch
            x_final = x_final.to(device=device, dtype=torch.float32)
            context = context.to(device=device, dtype=torch.float32)

            # Stochastic data augmentation: add small Gaussian jitter to targets
            if data_noise_std and data_noise_std > 0.0:
                x_final = x_final + torch.randn_like(x_final) * float(data_noise_std)

            optim.zero_grad(set_to_none=True)
            log_likelihood = compute_log_likelihood(model, x_final, context, steps_jitter)
            loss = -log_likelihood.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            running += float(loss.item())
            if progress_cb is not None:
                progress_cb(epoch, epochs, j + 1, total, float(loss.item()))

        scheduler.step()
        avg = running / max(1, len(dataloader))
        if ckpt_dir is not None:
            save_checkpoint(model, optim, f"{ckpt_dir}/cnf_epoch{epoch}.pt", epoch=epoch, loss=avg)

    return model
