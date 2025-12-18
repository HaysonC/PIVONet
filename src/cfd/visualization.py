"""Support utilities for comparing CFD results and animations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse
import numpy as np

from ..cfd.evaluation import ComparisonResult, compare_final_positions
from ..interfaces.visualization import PlotArtifact
from ..utils.paths import resolve_data_path
from ..utils.trajectory_io import load_trajectory_bundle


def _ellipse_from_gaussian(
    mean: np.ndarray, cov: np.ndarray, n_std: float = 2.0, **kwargs
) -> Ellipse:
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 1e-12))
    return Ellipse(
        xy=(float(mean[0]), float(mean[1])),
        width=float(width),
        height=float(height),
        angle=float(theta),
        fill=False,
        **kwargs,
    )


def _plot_flow_overlay(
    ax: Axes,
    flow_fn: Any,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    density: int = 20,
    mode: str = "quiver",
) -> None:
    if flow_fn is None:
        return
    xs = np.linspace(xlim[0], xlim[1], density)
    ys = np.linspace(ylim[0], ylim[1], density)
    X, Y = np.meshgrid(xs, ys)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            vector = flow_fn(np.array([X[i, j], Y[i, j]]))
            U[i, j] = vector[0]
            V[i, j] = vector[1] if len(vector) > 1 else 0.0
    if mode == "stream":
        ax.streamplot(X, Y, U, V, color="k", density=1.0, arrowsize=1.0, linewidth=0.8)
    else:
        ax.quiver(X, Y, U, V, color="k", alpha=0.8)


def _draw_varying_channel(
    ax: Axes, flow_fn: Any, xlim: tuple[float, float], steps: int = 200
) -> None:
    if flow_fn is None or not hasattr(flow_fn, "Hx"):
        return
    Hx = getattr(flow_fn, "Hx")
    xs = np.linspace(xlim[0], xlim[1], steps)
    top = [Hx(x) for x in xs]
    bot = [-Hx(x) for x in xs]
    ax.plot(xs, top, color="k", linestyle="--", linewidth=1.2, alpha=0.5)
    ax.plot(xs, bot, color="k", linestyle="--", linewidth=1.2, alpha=0.5)


def animation_to_html(
    anim: animation.FuncAnimation, embed_limit_mb: float | None = None
) -> str:
    if embed_limit_mb is not None:
        try:
            matplotlib.rcParams["animation.embed_limit"] = float(embed_limit_mb)
        except Exception:
            pass
    return anim.to_jshtml()


@dataclass(frozen=True)
class ComparisonArtifact:
    plot: PlotArtifact
    metrics: ComparisonResult


class CFDVisualizer:
    """Compare trajectory exports and produce diagnostics/animations."""

    def __init__(
        self, max_particles: int = 200, figsize: tuple[float, float] = (10, 4)
    ) -> None:
        self.max_particles = max_particles
        self.figsize = figsize

    def compare_trajectories(
        self,
        first: str | Path,
        second: str | Path,
        output_path: str | Path | None = None,
        title: str | None = None,
        flow_fn: Any | None = None,
        overlay_mode: str = "quiver",
        init_mean: np.ndarray | None = None,
        init_cov: np.ndarray | None = None,
        show: bool = False,
    ) -> ComparisonArtifact:
        bundle_a = load_trajectory_bundle(first)
        bundle_b = load_trajectory_bundle(second)
        history_a = bundle_a.history
        history_b = bundle_b.history

        fig, axes = plt.subplots(1, 2, figsize=self.figsize, dpi=140)
        axes_list = axes if isinstance(axes, (list, tuple)) else [axes]
        xlim_a = (float(history_a[:, :, 0].min()), float(history_a[:, :, 0].max()))
        ylim_a = (float(history_a[:, :, 1].min()), float(history_a[:, :, 1].max()))
        xlim_b = (float(history_b[:, :, 0].min()), float(history_b[:, :, 0].max()))
        ylim_b = (float(history_b[:, :, 1].min()), float(history_b[:, :, 1].max()))
        self._draw_history(
            axes_list[0],
            history_a,
            "reference",
            init_mean,
            init_cov,
            flow_fn,
            overlay_mode,
            xlim_a,
            ylim_a,
        )
        self._draw_history(
            axes_list[1],
            history_b,
            "comparison",
            init_mean,
            init_cov,
            flow_fn,
            overlay_mode,
            xlim_b,
            ylim_b,
        )

        for ax in axes_list:
            ax.set_aspect("equal")
            ax.grid(True, linestyle="--", alpha=0.4)

        fig.suptitle(title or "Trajectory comparison", fontsize=14)
        path = self._resolve_output_path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        if not show:
            plt.close(fig)

        metrics = compare_final_positions(
            bundle_a.final_positions, bundle_b.final_positions
        )
        return ComparisonArtifact(plot=PlotArtifact(path=path), metrics=metrics)

    def _draw_history(
        self,
        ax: Axes,
        history: np.ndarray,
        label: str,
        init_mean: np.ndarray | None,
        init_cov: np.ndarray | None,
        flow_fn: Any | None,
        overlay_mode: str,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
    ) -> None:
        indices = self._select_indices(history.shape[1])
        _plot_flow_overlay(ax, flow_fn, xlim, ylim, mode=overlay_mode)
        for idx in indices:
            ax.plot(history[:, idx, 0], history[:, idx, 1], alpha=0.6)
        if init_mean is not None and init_cov is not None:
            ell = _ellipse_from_gaussian(
                np.asarray(init_mean).reshape(2),
                np.asarray(init_cov).reshape(2, 2),
                edgecolor="red",
                linewidth=1.5,
            )
            ax.add_patch(ell)
            ax.scatter(
                [init_mean[0]], [init_mean[1]], color="red", s=20, label="Initial mean"
            )
        if (
            flow_fn is not None
            and hasattr(flow_fn, "boundary_type")
            and getattr(flow_fn, "boundary_type") == "varying-channel"
        ):
            _draw_varying_channel(ax, flow_fn, xlim)
        ax.set_title(label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    def plot_density_histogram(
        self,
        true_history: np.ndarray,
        pred_history: np.ndarray,
        output_path: str | Path | None = None,
        bins: int = 60,
        show: bool = False,
    ) -> PlotArtifact:
        fig, axes = plt.subplots(1, 2, figsize=self.figsize, dpi=140)
        axes_list = axes if isinstance(axes, (list, tuple)) else [axes]
        for ax in axes_list:
            ax.set_facecolor("white")
        axes_list[0].hist2d(
            true_history[:, -1, 0], true_history[:, -1, 1], bins=bins, cmap="viridis"
        )
        axes_list[0].set_title("True final positions")
        axes_list[1].hist2d(
            pred_history[:, -1, 0], pred_history[:, -1, 1], bins=bins, cmap="viridis"
        )
        axes_list[1].set_title("Predicted final positions")
        for ax in axes_list:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        fig.tight_layout()
        path = self._resolve_output_path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        if not show:
            plt.close(fig)
        return PlotArtifact(path=path)

    def animate_comparison(
        self,
        true_history: np.ndarray,
        pred_history: np.ndarray | None = None,
        flow_fn: Any | None = None,
        H: float | None = None,
        n_show: int = 50,
        interval: int = 30,
        tail: int | None = None,
        init_mean: np.ndarray | None = None,
        init_cov: np.ndarray | None = None,
        max_frames: int = 300,
        frame_stride: int | None = None,
        embed_limit_mb: float | None = None,
    ) -> animation.FuncAnimation:
        fig, ax = plt.subplots(figsize=self.figsize, dpi=140)
        ax.set_facecolor("white")
        candidates = true_history.shape[0]
        if pred_history is not None:
            candidates = min(candidates, pred_history.shape[0])
        idx = np.linspace(0, candidates - 1, min(n_show, candidates)).astype(int)

        T_true = true_history.shape[1]
        T_pred = pred_history.shape[1] if pred_history is not None else 0
        T = max(T_true, T_pred)

        xlim = (float(true_history[:, :, 0].min()), float(true_history[:, :, 0].max()))
        ylim = (float(true_history[:, :, 1].min()), float(true_history[:, :, 1].max()))
        if pred_history is not None:
            xlim = (
                min(xlim[0], float(pred_history[:, :, 0].min())),
                max(xlim[1], float(pred_history[:, :, 0].max())),
            )
            ylim = (
                min(ylim[0], float(pred_history[:, :, 1].min())),
                max(ylim[1], float(pred_history[:, :, 1].max())),
            )

        _plot_flow_overlay(ax, flow_fn, xlim, ylim)
        if init_mean is not None and init_cov is not None:
            ell = _ellipse_from_gaussian(
                np.asarray(init_mean).reshape(2),
                np.asarray(init_cov).reshape(2, 2),
                edgecolor="red",
                linewidth=1.5,
            )
            ax.add_patch(ell)
            ax.scatter([init_mean[0]], [init_mean[1]], color="red", s=20)

        lines_true = [
            ax.plot(
                [],
                [],
                color="tab:blue",
                alpha=0.85,
                linewidth=1.2,
                label=("True" if k == 0 else "_nolegend_"),
            )[0]
            for k in range(len(idx))
        ]
        lines_pred = None
        if pred_history is not None:
            lines_pred = [
                ax.plot(
                    [],
                    [],
                    color="tab:orange",
                    alpha=0.85,
                    linewidth=1.2,
                    label=("Pred" if k == 0 else "_nolegend_"),
                )[0]
                for k in range(len(idx))
            ]

        if H is not None:
            ax.axhline(H, color="k", linestyle="--", linewidth=1)
            ax.axhline(-H, color="k", linestyle="--", linewidth=1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("x (position)")
        ax.set_ylabel("y (position)")
        ax.set_title("Animated trajectories (true vs predicted)")

        def init():
            for line in lines_true:
                line.set_data([], [])
            if lines_pred is not None:
                for line in lines_pred:
                    line.set_data([], [])
            return (*lines_true, *(lines_pred or []))

        def update(frame: int):
            frame_true = min(frame, T_true - 1)
            frame_pred = min(frame, T_pred - 1) if pred_history is not None else 0
            start_true = 0 if tail is None else max(0, frame_true - tail)
            start_pred = 0 if tail is None else max(0, frame_pred - tail)
            for k, idx_i in enumerate(idx):
                lines_true[k].set_data(
                    true_history[idx_i, start_true : frame_true + 1, 0],
                    true_history[idx_i, start_true : frame_true + 1, 1],
                )
                if lines_pred is not None and pred_history is not None:
                    lines_pred[k].set_data(
                        pred_history[idx_i, start_pred : frame_pred + 1, 0],
                        pred_history[idx_i, start_pred : frame_pred + 1, 1],
                    )
            return (*lines_true, *(lines_pred or []))

        stride = frame_stride or max(1, int(max(1, T // max_frames)))
        frames = list(range(0, T, stride))
        anim = animation.FuncAnimation(
            fig, update, frames=frames, init_func=init, interval=interval, blit=True
        )
        if not embed_limit_mb:
            plt.close(fig)
        return anim

    def _select_indices(self, num_particles: int) -> list[int]:
        if num_particles <= self.max_particles:
            return list(range(num_particles))
        step = num_particles / self.max_particles
        return [int(i * step) for i in range(self.max_particles)]

    def _resolve_output_path(self, candidate: str | Path | None) -> Path:
        if candidate:
            path = Path(candidate)
        else:
            path = resolve_data_path("cfd", "plots", "comparison.png")
        if path.suffix == "":
            path = path.with_suffix(".png")
        return path
