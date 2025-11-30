"""Visualization helpers for training metrics and experiment diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..interfaces.visualization import PlotArtifact


def _resolve_output(path_like: str | Path) -> Path:
    path = Path(path_like)
    if not path.suffix:
        path = path.with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_loss_curve(losses: Sequence[float], output_path: str | Path, *, title: str | None = None) -> PlotArtifact:
    if not losses:
        raise ValueError("Cannot plot an empty loss history.")
    target = _resolve_output(output_path)
    steps = np.arange(1, len(losses) + 1)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
    ax.plot(steps, losses, color="#ff7f0e", linewidth=2.0)
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.set_title(title or "Training loss")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return PlotArtifact(path=target)


def plot_metric_grid(
    metric_history: Sequence[Mapping[str, float]],
    output_path: str | Path,
    *,
    metric_keys: Iterable[str] | None = None,
    title: str | None = None,
) -> PlotArtifact:
    if not metric_history:
        raise ValueError("Metric history is empty; nothing to plot.")
    keys = list(metric_keys) if metric_keys is not None else _infer_metric_keys(metric_history)
    if not keys:
        raise ValueError("No metric keys were provided or inferred from history.")
    target = _resolve_output(output_path)
    steps = np.arange(1, len(metric_history) + 1)
    rows = len(keys)
    fig, axes = plt.subplots(rows, 1, figsize=(7, 3 * rows), dpi=160, sharex=True)
    if rows == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        values = [entry.get(key, np.nan) for entry in metric_history]
        ax.plot(steps, values, linewidth=1.8, label=key)
        ax.set_ylabel(key)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.legend(loc="best")
    axes[-1].set_xlabel("training step")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return PlotArtifact(path=target)


def _infer_metric_keys(metric_history: Sequence[Mapping[str, float]]) -> list[str]:
    ignored = {"epoch", "step"}
    keys: set[str] = set()
    for entry in metric_history:
        for key in entry.keys():
            if key not in ignored:
                keys.add(key)
    return sorted(keys)
