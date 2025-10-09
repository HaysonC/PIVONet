from __future__ import annotations

import numpy as np
import torch
from scipy.stats import entropy


def mean_squared_displacement(trajs: np.ndarray) -> float:
    # trajs: (N, T+1, 2)
    disp = trajs[:, -1] - trajs[:, 0]
    return float(np.mean(np.sum(disp**2, axis=1)))


def kl_divergence_2d(p_samples: np.ndarray, q_samples: np.ndarray, bins: int = 50) -> float:
    # Estimate KL(P||Q) from histogram densities on final positions
    H_range = [
        [min(p_samples[:, 0].min(), q_samples[:, 0].min()), max(p_samples[:, 0].max(), q_samples[:, 0].max())],
        [min(p_samples[:, 1].min(), q_samples[:, 1].min()), max(p_samples[:, 1].max(), q_samples[:, 1].max())],
    ]
    Hp, xedges, yedges = np.histogram2d(p_samples[:, 0], p_samples[:, 1], bins=bins, range=H_range, density=True)
    Hq, _, _ = np.histogram2d(q_samples[:, 0], q_samples[:, 1], bins=[xedges, yedges], density=True)
    p = Hp.flatten() + 1e-12
    q = Hq.flatten() + 1e-12
    return float(entropy(p, q))


def overlap_ratio(p_samples: np.ndarray, q_samples: np.ndarray, eps: float = 0.05) -> float:
    # Percentage of p points that have at least one q within eps (Euclidean)
    from scipy.spatial import KDTree

    tree = KDTree(q_samples)
    counts = tree.query_ball_point(p_samples, r=eps)
    hits = sum(1 for c in counts if len(c) > 0)
    return float(hits / len(p_samples))
