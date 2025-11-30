"""Modeling-specific configuration shared between training helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelingConfig:
    """Hyperparameter bundle used by the diffusion encoder + CNF training stack."""

    run_name: str | None = None
    cache_subdir: str = "modeling"
    latent_dim: int = 16
    context_dim: int = 64
    encoder_lr: float = 1e-3
    encoder_steps: int = 8
    cnf_lr: float = 2e-4
    cnf_steps: int = 6
    cnf_hidden_dim: int = 128
