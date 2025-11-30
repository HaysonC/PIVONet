"""Visualization helpers for raw velocity field snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..interfaces.visualization import PlotArtifact
from ..utils.paths import resolve_data_path


class VelocityFieldPlotter:
    """Render velocity vector statistics from ``.npy`` snapshots."""

    def __init__(
        self,
        sample_points: int = 50_000,
        cmap: str = "viridis",
        figsize: Tuple[float, float] = (8, 4),
        random_seed: int | None = None,
    ) -> None:
        self.sample_points = sample_points
        self.cmap = plt.get_cmap(cmap)
        self.figsize = figsize
        self.rng = np.random.default_rng(random_seed)

    def plot_from_file(
        self,
        velocity_path: str | Path,
        output_path: str | Path | None = None,
        show: bool = False,
        title: str | None = None,
    ) -> PlotArtifact:
        field = self._load_field(velocity_path)
        return self.plot(field, output_path=output_path, show=show, title=title)

    def plot(
        self,
        field: np.ndarray,
        output_path: str | Path | None = None,
        show: bool = False,
        title: str | None = None,
    ) -> PlotArtifact:
        planar = self._sanitize_field(field)
        sample = self._subsample(planar)
        magnitudes = np.linalg.norm(sample, axis=1)

        fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=self.figsize, dpi=160)

        scatter = ax_scatter.scatter(
            sample[:, 0],
            sample[:, 1],
            c=magnitudes,
            cmap=self.cmap,
            s=3,
            alpha=0.6,
        )
        ax_scatter.set_xlabel("vx")
        ax_scatter.set_ylabel("vy")
        ax_scatter.set_title(title or "Velocity phase space")
        fig.colorbar(scatter, ax=ax_scatter, label="|v|")
        ax_scatter.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)

        ax_hist.hist(magnitudes, bins=60, color="#1f77b4", alpha=0.85)
        ax_hist.set_xlabel("|v|")
        ax_hist.set_ylabel("count")
        ax_hist.set_title("Speed distribution")

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

    def _sanitize_field(self, field: np.ndarray) -> np.ndarray:
        array = np.asarray(field)
        if array.ndim == 1:
            if array.size % 2 != 0:
                raise ValueError("Cannot reshape velocity array of odd length")
            array = array.reshape(-1, 2)
        if array.ndim != 2 or array.shape[1] < 2:
            raise ValueError("Velocity field must be shape (N, >=2)")
        return array[:, :2]

    def _subsample(self, array: np.ndarray) -> np.ndarray:
        if array.shape[0] <= self.sample_points:
            return array
        indices = self.rng.choice(array.shape[0], size=self.sample_points, replace=False)
        return array[indices]

    def _resolve_output_path(self, output_path: str | Path) -> Path:
        path = Path(output_path)
        if not path.is_absolute():
            path = resolve_data_path(*path.parts)
        if path.suffix == "":
            path = path.with_suffix(".png")
        return path

    def _load_field(self, velocity_path: str | Path) -> np.ndarray:
        path = Path(velocity_path)
        if not path.is_absolute():
            path = resolve_data_path(*path.parts)
        if not path.exists():
            raise FileNotFoundError(f"Velocity snapshot not found: {path}")
        return np.load(path, allow_pickle=False)
