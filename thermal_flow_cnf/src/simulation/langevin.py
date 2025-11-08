from __future__ import annotations

import os
from typing import Callable, Tuple, Optional

import numpy as np
from tqdm import trange

from .boundary import reflect_y, no_slip_damping


def simulate_trajectory(
    flow_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    D: float,
    dt: float,
    T: int,
    H: float,
    theta: float | None = None,
    seed: int | None = None,
) -> Tuple[np.ndarray, float | None, np.ndarray]:
    """
    Simulate one 2D trajectory with overdamped Langevin dynamics and reflecting y-boundaries.

    Args:
        flow_fn: callable returning velocity u(x, y) -> np.array([ux, uy])
        x0: initial 2D position (array-like shape (2,))
        D: diffusion coefficient
        dt: time step
        T: total number of steps (int)
        H: boundary half-height (reflect at y = ±H)
        theta: optional scalar condition parameter recorded with trajectory
        seed: optional RNG seed

    Returns:
        (x0, theta, traj) where traj has shape (T+1, 2)
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    x = np.array(x0, dtype=float).reshape(2)
    traj = np.zeros((int(T) + 1, 2), dtype=float)
    traj[0] = x

    sigma = np.sqrt(2.0 * float(D) * float(dt))

    for t in range(1, int(T) + 1):
        u = np.array(flow_fn(x), dtype=float).reshape(2)
        # Apply smooth no-slip damping near channel walls in advective component
        damp = no_slip_damping(x[1], H)
        u = u * float(damp)
        noise = sigma * rng.standard_normal(2)
        x = x + u * dt + noise

        # Reflect only in y-direction at ±H
        y_reflected, bounced = reflect_y(x[1], H)
        if bounced:
            # For a simple elastic reflection, flip the y-step; here we just set y
            x[1] = y_reflected
        traj[t] = x

    return np.array(x0, dtype=float).reshape(2), theta, traj


def simulate_dataset(
    flow_fn: Callable[[np.ndarray], np.ndarray],
    num_particles: int,
    D: float,
    dt: float,
    T: int,
    H: float,
    x0_sampler: Callable[[int], np.ndarray] | None = None,
    init_dist: str = "uniform",
    init_mean: Optional[Tuple[float, float]] = None,
    init_cov: Optional[np.ndarray] = None,
    theta: float | None = None,
    save_dir: str | None = None,
    prefix: str = "sim",
    seed: int | None = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
):
    """Simulate a dataset of trajectories and optionally save them as .npz.

    Saves fields: x0s (N,2), thetas (N,), trajs (N,T+1,2)
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    if x0_sampler is None:
        def default_uniform(n: int) -> np.ndarray:
            xs = rng.uniform(0.0, 1.0, size=(n,))
            ys = rng.uniform(-H, H, size=(n,))
            return np.stack([xs, ys], axis=1)
        def default_gaussian(n: int) -> np.ndarray:
            mu = np.array(init_mean if init_mean is not None else [0.0, 0.0], dtype=float)
            cov = np.array(init_cov if init_cov is not None else [[0.1, 0.0],[0.0, 0.1]], dtype=float)
            pts = rng.multivariate_normal(mu, cov, size=n)
            # clamp y within bounds to start inside channel
            pts[:, 1] = np.clip(pts[:, 1], -H, H)
            return pts
        sampler = default_uniform if init_dist == "uniform" else default_gaussian
    else:
        sampler = x0_sampler

    x0s = sampler(num_particles)
    thetas = np.full((num_particles,), theta if theta is not None else 0.0, dtype=float)
    trajs = np.zeros((num_particles, int(T) + 1, 2), dtype=float)

    iterator = range(num_particles) if progress_cb is not None else trange(num_particles, desc="Simulating")
    for i in iterator:
        x0_i = x0s[i]
        _, theta_i, traj_i = simulate_trajectory(flow_fn, x0_i, D, dt, T, H, theta)
        thetas[i] = theta if theta is not None else 0.0
        trajs[i] = traj_i
        if progress_cb is not None:
            progress_cb(i + 1, num_particles)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{prefix}_N{num_particles}_T{T}.npz")
        meta = {
            "x0s": x0s,
            "thetas": thetas,
            "trajs": trajs,
            "dt": dt,
            "D": D,
            "H": H,
            "init_dist": init_dist,
            "init_mean": np.array(init_mean, dtype=float) if init_mean is not None else np.array([np.mean(x0s[:,0]), np.mean(x0s[:,1])], dtype=float),
            "init_cov": np.cov(x0s.T),
        }
        np.savez_compressed(out_path, **meta)
        return out_path

    return {"x0s": x0s, "thetas": thetas, "trajs": trajs, "dt": dt, "D": D, "H": H, "init_dist": init_dist, "init_mean": init_mean, "init_cov": np.cov(x0s.T)}
