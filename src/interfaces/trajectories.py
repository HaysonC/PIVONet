"""Shared data structures for particle trajectory simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class TrajectoryResult:
    """Container for simulated particle trajectories."""

    history: np.ndarray  # shape: (num_steps + 1, num_particles, 2)
    timesteps: Sequence[float]

    @property
    def final_positions(self) -> np.ndarray:
        return self.history[-1]
    
    @property
    def num_particles(self) -> int:
        return self.history.shape[1]
