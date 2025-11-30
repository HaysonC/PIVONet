"""Lightweight neural tooling for CFD modeling."""

from __future__ import annotations

from .cnf import CNFModel
from .encoder import DiffusionEncoderNet
from .mlp import MLP

__all__ = ["CNFModel", "DiffusionEncoderNet", "MLP"]
