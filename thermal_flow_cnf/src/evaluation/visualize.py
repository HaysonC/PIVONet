from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_trajectories(trajs: np.ndarray, flow_fn=None, H: float | None = None, n_show: int = 50):
    plt.figure(figsize=(6, 4))
    N = min(n_show, trajs.shape[0])
    idx = np.linspace(0, trajs.shape[0] - 1, N).astype(int)
    for i in idx:
        plt.plot(trajs[i, :, 0], trajs[i, :, 1], alpha=0.5)
    if H is not None:
        plt.axhline(H, color='k', linestyle='--', linewidth=1)
        plt.axhline(-H, color='k', linestyle='--', linewidth=1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trajectories')
    plt.tight_layout()
    plt.show()


def plot_density_hist2d(samples_true: np.ndarray, samples_pred: np.ndarray, bins: int = 60):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist2d(samples_true[:, 0], samples_true[:, 1], bins=bins, cmap='viridis')
    axes[0].set_title('True final positions')
    axes[1].hist2d(samples_pred[:, 0], samples_pred[:, 1], bins=bins, cmap='viridis')
    axes[1].set_title('Predicted final positions')
    for ax in axes:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    plt.tight_layout()
    plt.show()
