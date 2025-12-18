"""Hybrid modeling stack that pairs variational diffusion encoding with a CNF advection model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from torch import optim

from ..interfaces.cfd import ModelingResult, TrajectoryExport
from ..interfaces.modeling import ModelingConfig
from ..interfaces.trajectories import TrajectoryResult
from ..networks.cnf import CNFModel
from ..networks.encoder import DiffusionEncoderNet


@dataclass(frozen=True)
class EncoderTrainingState:
    loss: float
    context: torch.Tensor
    state_path: Path


@dataclass(frozen=True)
class CNFTrainingState:
    loss: float
    state_path: Path


class DiffusionEncoder:
    """Variational encoder that summarizes trajectories into a latent context vector."""

    def __init__(
        self,
        cache_dir: Path,
        latent_dim: int = 16,
        context_dim: int = 64,
        lr: float = 1e-3,
        steps: int = 8,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = DiffusionEncoderNet(z_dim=latent_dim, ctx_dim=context_dim).to(
            self.device
        )
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.cache_dir = cache_dir
        self.steps = max(steps, 1)

    def _build_dataset(
        self, trajectory: TrajectoryResult
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history = torch.from_numpy(trajectory.history).float().to(self.device)
        sequence = history.permute(1, 0, 2)
        time_grid = torch.linspace(
            0.0, 1.0, steps=sequence.shape[1], device=self.device
        )
        return sequence, time_grid

    def _kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def train(self, trajectory: TrajectoryResult) -> EncoderTrainingState:
        sequence, time_grid = self._build_dataset(trajectory)
        batch_size = sequence.shape[0]
        time_batch = time_grid.unsqueeze(0).expand(batch_size, -1)
        loss = torch.tensor(0.0, device=self.device)
        context = torch.zeros(
            batch_size, self.net.encoder.ctx_proj.out_features, device=self.device
        )
        for _ in range(self.steps):
            self.optimizer.zero_grad()
            context, mu, logvar = self.net(sequence, time_batch)
            loss = self._kl_loss(mu, logvar)
            loss.backward()
            self.optimizer.step()
        state_path = self.cache_dir / "encoder.pt"
        torch.save(self.net.state_dict(), state_path)
        return EncoderTrainingState(
            loss=loss.item(), context=context.detach().cpu(), state_path=state_path
        )


class TrajectoryCNF:
    """CNF that learns the advection dynamics of particle endpoints."""

    def __init__(
        self,
        cache_dir: Path,
        cond_dim: int = 64,
        hidden_dim: int = 128,
        lr: float = 2e-4,
        steps: int = 6,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNFModel(dim=2, cond_dim=cond_dim, hidden_dim=hidden_dim).to(
            self.device
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.cache_dir = cache_dir
        self.steps = max(steps, 1)

    def train(
        self, trajectory: TrajectoryResult, context: torch.Tensor
    ) -> CNFTrainingState:
        final_positions = (
            torch.from_numpy(trajectory.history[-1]).float().to(self.device)
        )
        context = context.to(self.device)
        loss = torch.tensor(0.0, device=self.device)
        for _ in range(self.steps):
            self.optimizer.zero_grad()
            log_prob = self.model.log_prob(final_positions, context)
            loss = -log_prob.mean()
            loss.backward()
            self.optimizer.step()
        state_path = self.cache_dir / "cnf.pt"
        torch.save(self.model.state_dict(), state_path)
        return CNFTrainingState(loss=loss.item(), state_path=state_path)


class HybridModel:
    """Coordinator that fits the encoder and CNF in sequence."""

    def __init__(self, cache_dir: Path, config: ModelingConfig | None = None) -> None:
        self.cache_dir = cache_dir
        cfg = config or ModelingConfig()
        self.encoder = DiffusionEncoder(
            cache_dir=cache_dir,
            latent_dim=cfg.latent_dim,
            context_dim=cfg.context_dim,
            lr=cfg.encoder_lr,
            steps=cfg.encoder_steps,
        )
        self.cnf = TrajectoryCNF(
            cache_dir=cache_dir,
            cond_dim=cfg.context_dim,
            hidden_dim=cfg.cnf_hidden_dim,
            lr=cfg.cnf_lr,
            steps=cfg.cnf_steps,
        )

    def fit(self, trajectory: TrajectoryExport | None) -> ModelingResult:
        encoder_state: EncoderTrainingState | None = None
        cnf_state: CNFTrainingState | None = None
        metrics: Dict[str, float] = {}

        if trajectory is not None:
            encoder_state = self.encoder.train(trajectory.result)
            cnf_state = self.cnf.train(trajectory.result, encoder_state.context)
            metrics["encoder_loss"] = encoder_state.loss
            metrics["cnf_loss"] = cnf_state.loss
            metrics["steps"] = float(trajectory.steps)

        return ModelingResult(
            encoder_state=encoder_state.state_path if encoder_state else None,
            cnf_state=cnf_state.state_path if cnf_state else None,
            metrics=metrics,
        )
