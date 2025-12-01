"""Training loops for CNF and variational SDE models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ..networks.cnf import CNFModel
from ..networks.variational_sde import VariationalSDEModel


@dataclass
class TrainingArtifacts:
    last_checkpoint: Path | None
    loss_history: list[float] = field(default_factory=list)
    metric_history: list[dict[str, float]] = field(default_factory=list)
    val_loss_history: list[float] = field(default_factory=list)
    val_metric_history: list[dict[str, float]] = field(default_factory=list)
    test_metric_history: list[dict[str, float]] = field(default_factory=list)


def train_cnf_model(
    model: CNFModel,
    dataloader: DataLoader,
    *,
    device: str = "cpu",
    epochs: int = 5,
    lr: float = 1e-3,
    ckpt_dir: Path | None = None,
    val_loader: DataLoader | None = None,
    test_loader: DataLoader | None = None,
    progress_cb: Optional[Callable[[str, int, int, int, int, float], None]] = None,
) -> TrainingArtifacts:
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    ckpt_path: Path | None = None
    loss_history: list[float] = []
    metric_history: list[dict[str, float]] = []
    val_loss_history: list[float] = []
    val_metric_history: list[dict[str, float]] = []
    test_metric_history: list[dict[str, float]] = []

    def _capture_progress(phase: str, epoch: int, total_epochs: int, step: int, total_steps: int, loss_value: float) -> None:
        if progress_cb is not None:
            progress_cb(phase, epoch, total_epochs, step, total_steps, loss_value)

    @torch.no_grad()
    def _eval_phase(loader: DataLoader | None, phase: str, epoch: int | None) -> None:
        if loader is None:
            return
        total = len(loader)
        if total == 0:
            return
        model.eval()
        for step, batch in enumerate(loader, start=1):
            x_final, _x0, _theta, context = batch
            x_final = x_final.to(device=device, dtype=torch.float32)
            context = context.to(device=device, dtype=torch.float32)
            # Defensive check: ensure inputs are finite before calling model
            if not torch.isfinite(x_final).all() or not torch.isfinite(context).all():
                raise ValueError(
                    "Encountered non-finite values in training batch. "
                    "Check trajectory bundles for NaNs/Infs. "
                    f"Batch info: step={step}, loader_len={total}"
                )
            log_prob = model.log_prob(x_final, context)
            loss = -log_prob.mean()
            loss_value = float(loss.item())
            record = {
                "step": float(step),
                "loss": loss_value,
            }
            if epoch is not None:
                record["epoch"] = float(epoch)
            _capture_progress(phase, epoch or 0, epochs, step, total, loss_value)
            if phase == "val":
                val_loss_history.append(loss_value)
                val_metric_history.append(record)
            elif phase == "test":
                test_metric_history.append(record)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        total = len(dataloader)
        for step, batch in enumerate(dataloader, start=1):
            x_final, _x0, _theta, context = batch
            x_final = x_final.to(device=device, dtype=torch.float32)
            context = context.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            log_prob = model.log_prob(x_final, context)
            loss = -log_prob.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += float(loss.item())
            loss_value = float(loss.item())
            loss_history.append(loss_value)
            metric_history.append(
                {
                    "epoch": float(epoch),
                    "step": float(step),
                    "loss": loss_value,
                }
            )
            _capture_progress("train", epoch, epochs, step, total, loss_value)
        scheduler.step()
        if ckpt_dir is not None:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"cnf_epoch{epoch}.pt"
            torch.save(model.state_dict(), ckpt_path)
        _eval_phase(val_loader, "val", epoch)
    _eval_phase(test_loader, "test", epochs)
    return TrainingArtifacts(
        last_checkpoint=ckpt_path,
        loss_history=loss_history,
        metric_history=metric_history,
        val_loss_history=val_loss_history,
        val_metric_history=val_metric_history,
        test_metric_history=test_metric_history,
    )


def train_variational_sde_model(
    model: VariationalSDEModel,
    dataloader: DataLoader,
    *,
    device: str = "cpu",
    epochs: int = 3,
    lr: float = 1e-3,
    ckpt_dir: Path | None = None,
    n_particles: int = 4,
    obs_std: float = 0.05,
    kl_warmup: float = 1.0,
    control_cost_scale: float = 1.0,
    n_integration_steps: int = 50,
    progress_cb: Optional[Callable[..., None]] = None,
) -> TrainingArtifacts:
    model = model.to(device)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    ckpt_path: Path | None = None
    loss_history: list[float] = []
    metric_history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        total = len(dataloader)
        for step, batch in enumerate(dataloader, start=1):
            traj, times, context, mask = batch
            traj = traj.to(device=device, dtype=torch.float32)
            times = times.to(device=device, dtype=torch.float32)
            context = context.to(device=device, dtype=torch.float32)
            mask = mask.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss, stats = model.compute_elbo(
                traj,
                times,
                context=context,
                mask=mask,
                n_particles=n_particles,
                obs_std=obs_std,
                kl_warmup=kl_warmup,
                control_cost_scale=control_cost_scale,
                n_integration_steps=n_integration_steps,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += float(loss.item())
            loss_value = float(loss.item())
            record = {"epoch": float(epoch), "step": float(step), "loss": loss_value}
            for key, value in (stats or {}).items():
                record[key] = float(value)
            loss_history.append(loss_value)
            metric_history.append(record)
            if progress_cb is not None:
                try:
                    progress_cb(epoch, epochs, step, total, loss_value, stats)
                except TypeError:
                    progress_cb(epoch, epochs, step, total, loss_value)
        scheduler.step()
        if ckpt_dir is not None:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"vsde_epoch{epoch}.pt"
            torch.save(model.state_dict(), ckpt_path)
    return TrainingArtifacts(last_checkpoint=ckpt_path, loss_history=loss_history, metric_history=metric_history)
