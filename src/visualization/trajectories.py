"""Visualization helpers for particle trajectories."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # ensure headless-friendly backend

import matplotlib.pyplot as plt
import numpy as np

from ..interfaces.trajectories import TrajectoryResult
from ..interfaces.visualization import PlotArtifact, TrajectoryVisualizer
from ..utils.paths import resolve_data_path


class TrajectoryPlotter(TrajectoryVisualizer):
    """Static plotter for particle trajectories."""

    def __init__(
        self,
        max_particles: int = 200,
        cmap: str = "plasma",
        figsize: tuple[float, float] = (6, 4),
    ) -> None:
        self.max_particles = max_particles
        self.cmap = plt.get_cmap(cmap)
        self.figsize = figsize

    def plot(
        self,
        trajectories: TrajectoryResult,
        output_path: Path | str | None = None,
        show: bool = False,
        title: str | None = None,
        **_: object,
    ) -> PlotArtifact:
        history = trajectories.history
        if history.ndim != 3 or history.shape[2] < 2:
            raise ValueError("Trajectory history must have shape (T, N, 2)")

        num_particles = history.shape[1]
        indices = self._select_indices(num_particles)
        colors = self.cmap(np.linspace(0, 1, len(indices)))

        fig, ax = plt.subplots(figsize=self.figsize, dpi=160)
        for color, idx in zip(colors, indices):
            xs = history[:, idx, 0]
            ys = history[:, idx, 1]
            ax.plot(xs, ys, color=color, linewidth=1.0, alpha=0.8)
            ax.scatter(xs[0], ys[0], color=color, s=10, marker="o", alpha=0.9)
            ax.scatter(xs[-1], ys[-1], color=color, s=14, marker="x", alpha=0.9)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(
            title or f"Trajectories ({len(indices)} of {num_particles} particles)"
        )
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)

        saved_path: Path | None = None
        if output_path is not None:
            saved_path = self._resolve_output_path(output_path)
            saved_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(saved_path, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return PlotArtifact(path=saved_path)

    def _resolve_output_path(self, output_path: Path | str) -> Path:
        path = Path(output_path)
        if not path.is_absolute():
            path = resolve_data_path(*path.parts)
        if path.suffix == "":
            path = path.with_suffix(".png")
        return path

    def _select_indices(self, num_particles: int) -> np.ndarray:
        if num_particles <= self.max_particles:
            return np.arange(num_particles)
        step = num_particles / self.max_particles
        return np.floor(np.arange(self.max_particles) * step).astype(int)
