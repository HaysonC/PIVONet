from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib as mpl
from matplotlib.patches import Ellipse


def _plot_flow_overlay(
    ax, flow_fn, xlim, ylim, density: int = 20, mode: str = "quiver"
):
    if flow_fn is None:
        return
    xs = np.linspace(xlim[0], xlim[1], density)
    ys = np.linspace(ylim[0], ylim[1], density)
    X, Y = np.meshgrid(xs, ys)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            u = flow_fn(np.array([X[i, j], Y[i, j]]))
            U[i, j] = u[0]
            V[i, j] = u[1] if len(u) > 1 else 0.0
    speed = np.sqrt(U**2 + V**2)
    if mode == "stream":
        # Use streamplot colored by speed, but render in black tones
        strm = ax.streamplot(
            X, Y, U, V, color="k", density=1.0, arrowsize=1.0, linewidth=0.8
        )
    else:
        # black arrows for quiver
        ax.quiver(X, Y, U, V, color="k", alpha=0.9)


def _ellipse_from_gaussian(
    mean: np.ndarray, cov: np.ndarray, n_std: float = 2.0, **kwargs
):
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


def _draw_varying_channel(
    ax, flow_fn, xlim, steps: int = 200, color: str = "k", alpha: float = 0.5
):
    """Draw top and bottom boundaries if flow_fn exposes Hx (varying channel)."""
    if flow_fn is None or not hasattr(flow_fn, "Hx"):
        return
    Hx = getattr(flow_fn, "Hx")
    xs = np.linspace(xlim[0], xlim[1], steps)
    top = [Hx(x) for x in xs]
    bot = [-Hx(x) for x in xs]
    ax.plot(xs, top, color="k", linestyle="--", linewidth=1.2, alpha=alpha)
    ax.plot(xs, bot, color="k", linestyle="--", linewidth=1.2, alpha=alpha)


def plot_trajectories(
    trajs: np.ndarray,
    flow_fn=None,
    H: float | None = None,
    n_show: int = 50,
    init_mean: np.ndarray | None = None,
    init_cov: np.ndarray | None = None,
    flow_overlay_mode: str = "quiver",
):
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    N = min(n_show, trajs.shape[0])
    idx = np.linspace(0, trajs.shape[0] - 1, N).astype(int)
    xs = trajs[:, :, 0]
    ys = trajs[:, :, 1]
    xlim = [float(xs.min()), float(xs.max())]
    ylim = [float(ys.min()), float(ys.max())]
    _plot_flow_overlay(ax, flow_fn, xlim, ylim, mode=flow_overlay_mode)
    for i in idx:
        ax.plot(trajs[i, :, 0], trajs[i, :, 1], alpha=0.6)
    if init_mean is not None and init_cov is not None:
        ell = _ellipse_from_gaussian(
            np.asarray(init_mean).reshape(2),
            np.asarray(init_cov).reshape(2, 2),
            n_std=2.0,
            edgecolor="red",
            linewidth=1.5,
        )
        ax.add_patch(ell)
        ax.scatter(
            [init_mean[0]], [init_mean[1]], color="red", s=20, label="Initial mean"
        )
    if (
        hasattr(flow_fn, "boundary_type")
        and getattr(flow_fn, "boundary_type") == "varying-channel"
    ):
        _draw_varying_channel(ax, flow_fn, xlim)
    elif H is not None:
        ax.axhline(H, color="k", linestyle="--", linewidth=1)
        ax.axhline(-H, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("x (position)")
    ax.set_ylabel("y (position)")
    ax.set_title("Trajectories with flow field and initial distribution")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_density_hist2d(
    samples_true: np.ndarray, samples_pred: np.ndarray, bins: int = 60
):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")
    axes[0].hist2d(samples_true[:, 0], samples_true[:, 1], bins=bins, cmap="viridis")
    axes[0].set_title("True final positions")
    axes[1].hist2d(samples_pred[:, 0], samples_pred[:, 1], bins=bins, cmap="viridis")
    axes[1].set_title("Predicted final positions")
    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.tight_layout()
    return fig


def animate_trajectories(
    trajs_true: np.ndarray,
    trajs_pred: np.ndarray | None = None,
    flow_fn=None,
    H: float | None = None,
    n_show: int = 50,
    interval: int = 30,
    tail: int | None = None,
    init_mean: np.ndarray | None = None,
    init_cov: np.ndarray | None = None,
    max_frames: int = 300,
    frame_stride: int | None = None,
):
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    n_true = trajs_true.shape[0]
    n_pred = trajs_pred.shape[0] if trajs_pred is not None else None
    if trajs_pred is not None:
        N = int(min(n_show, n_true, n_pred))
        max_idx = int(min(n_true, n_pred) - 1)
    else:
        N = int(min(n_show, n_true))
        max_idx = int(n_true - 1)
    idx = np.linspace(0, max_idx, N).astype(int)
    T_true = trajs_true.shape[1]
    T_pred = trajs_pred.shape[1] if trajs_pred is not None else 0
    T = int(max(T_true, T_pred))

    # Compute limits without concatenating mismatched time dims
    x_min = float(trajs_true[:, :, 0].min())
    x_max = float(trajs_true[:, :, 0].max())
    y_min = float(trajs_true[:, :, 1].min())
    y_max = float(trajs_true[:, :, 1].max())
    if trajs_pred is not None:
        x_min = min(x_min, float(trajs_pred[:, :, 0].min()))
        x_max = max(x_max, float(trajs_pred[:, :, 0].max()))
        y_min = min(y_min, float(trajs_pred[:, :, 1].min()))
        y_max = max(y_max, float(trajs_pred[:, :, 1].max()))
    xlim = [x_min, x_max]
    ylim = [y_min, y_max]
    _plot_flow_overlay(ax, flow_fn, xlim, ylim)
    if init_mean is not None and init_cov is not None:
        ell = _ellipse_from_gaussian(
            np.asarray(init_mean).reshape(2),
            np.asarray(init_cov).reshape(2, 2),
            n_std=2.0,
            edgecolor="red",
            linewidth=1.5,
        )
        ax.add_patch(ell)
        ax.scatter(
            [init_mean[0]], [init_mean[1]], color="red", s=20, label="Initial mean"
        )

    # Create line objects; only first gets legend label to avoid duplicates
    lines_true = []
    for k, _ in enumerate(idx):
        lbl = "True" if k == 0 else "_nolegend_"
        lines_true.append(
            ax.plot([], [], color="tab:blue", alpha=0.85, linewidth=1.2, label=lbl)[0]
        )
    lines_pred = None
    if trajs_pred is not None:
        lines_pred = []
        for k, _ in enumerate(idx):
            lbl = "Pred" if k == 0 else "_nolegend_"
            lines_pred.append(
                ax.plot(
                    [], [], color="tab:orange", alpha=0.85, linewidth=1.2, label=lbl
                )[0]
            )

    if (
        hasattr(flow_fn, "boundary_type")
        and getattr(flow_fn, "boundary_type") == "varying-channel"
    ):
        _draw_varying_channel(ax, flow_fn, xlim)
    elif H is not None:
        ax.axhline(H, color="k", linestyle="--", linewidth=1)
        ax.axhline(-H, color="k", linestyle="--", linewidth=1)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlabel("x (position)")
    ax.set_ylabel("y (position)")
    ax.set_title("Animated Trajectories (true vs model) — normalization and spreading")
    ax.legend(loc="upper right", fontsize=8)

    def init():
        for ln in lines_true:
            ln.set_data([], [])
        if lines_pred is not None:
            for ln in lines_pred:
                ln.set_data([], [])
        return (*lines_true, *(lines_pred or []))

    def update(frame):
        # Clamp frame separately for true and predicted to prevent disappearance when one sequence is shorter
        frame_true = int(min(frame, T_true - 1))
        frame_pred = int(min(frame, T_pred - 1)) if trajs_pred is not None else 0
        # Tail management
        if tail is None:
            start_true = 0
            start_pred = 0
        else:
            start_true = max(0, frame_true - tail)
            start_pred = max(0, frame_pred - tail)
        for k, i in enumerate(idx):
            # True trajectory slice
            lines_true[k].set_data(
                trajs_true[i, start_true : frame_true + 1, 0],
                trajs_true[i, start_true : frame_true + 1, 1],
            )
            # Predicted trajectory slice (if available)
            if lines_pred is not None and trajs_pred is not None:
                lines_pred[k].set_data(
                    trajs_pred[i, start_pred : frame_pred + 1, 0],
                    trajs_pred[i, start_pred : frame_pred + 1, 1],
                )
        return (*lines_true, *(lines_pred or []))

    # Determine frames sequence to limit total frames for embedding size
    stride = (
        int(frame_stride)
        if frame_stride is not None
        else max(1, int(np.ceil(T / max(1, max_frames))))
    )
    frames_seq = list(range(0, T, stride))
    # Rebuild legend once (duplicate labels suppressed by '_nolegend_')
    handles, labels = ax.get_legend_handles_labels()
    uniq = []
    seen = set()
    for h, lb in zip(handles, labels):
        if lb == "_nolegend_" or lb in seen:
            continue
        seen.add(lb)
        uniq.append((h, lb))
    if uniq:
        ax.legend(
            [h for h, _ in uniq], [lb for _, lb in uniq], loc="upper right", fontsize=8
        )
    anim = animation.FuncAnimation(
        fig, update, frames=frames_seq, init_func=init, interval=interval, blit=True
    )
    return anim


def animation_to_html(
    anim: animation.FuncAnimation, embed_limit_mb: float | None = None
) -> str:
    """Convert a Matplotlib animation to embeddable HTML using JS.

    Optionally set a higher or lower embed_limit (in MB) to control the maximum embedded size.
    """
    if embed_limit_mb is not None:
        try:
            mpl.rcParams["animation.embed_limit"] = float(embed_limit_mb)
        except Exception:
            pass
    return anim.to_jshtml()
