"""Particle trajectory generation utilities."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from ..interfaces.data_sources import VelocityFieldSource
from ..interfaces.trajectories import TrajectoryResult


KDTreeType = Any

try:
    _scipy_spatial = import_module("scipy.spatial")
    _KDTreeClass = getattr(_scipy_spatial, "cKDTree", None)
except Exception:  # pragma: no cover - optional dependency
    _KDTreeClass = None


class ParticleTrajectorySimulator:
    """Simulate particles advected by velocity fields with diffusion."""

    def __init__(
        self,
        velocity_source: VelocityFieldSource,
        diffusion_coefficient: float,
        dt: float = 1e-2,
        seed: int | None = None,
        respect_snapshot_timing: bool = True,
    ) -> None:
        self.velocity_source = velocity_source
        self.diffusion = float(diffusion_coefficient)
        self.dt = dt
        self.respect_snapshot_timing = respect_snapshot_timing
        self.rng = np.random.default_rng(seed)

        self._points: np.ndarray | None = None
        self._kdt: KDTreeType | None = None

        velocity_dir = getattr(velocity_source, "_velocity_dir", None)
        if velocity_dir is not None:
            points_candidates = [
                Path(velocity_dir) / "points.npy",
                Path(velocity_dir) / "mesh_points.npy",
                Path(velocity_dir).parent / "points.npy",
                Path(velocity_dir).parent / "mesh_points.npy",
            ]
            for candidate in points_candidates:
                try:
                    if candidate.exists():
                        pts = np.load(candidate, allow_pickle=False)
                        if pts.ndim == 2 and pts.shape[1] >= 2:
                            self._points = pts[:, :2].astype(float)
                            break
                except Exception:  # pragma: no cover - best effort
                    continue
        if self._points is not None and _KDTreeClass is not None:
            self._kdt = _KDTreeClass(self._points)

    def simulate(
        self,
        num_particles: int,
        max_steps: int | None = None,
    ) -> TrajectoryResult:
        """Run the trajectory simulation.

        Args:
            num_particles: Number of particles ``n`` to track.
            max_steps: Optional cap on the number of velocity snapshots to use.

        Returns:
            :class:`TrajectoryResult` containing the trajectory history.
        """

        if num_particles <= 0:
            raise ValueError("num_particles must be positive")

        positions = self._initialize_particles(num_particles)
        history = [positions.copy()]
        timesteps = [0.0]

        previous_time: float | None = timesteps[0]

        for step_index, (snapshot_time, field) in enumerate(
            self.velocity_source.iter_velocity_fields()
        ):
            if max_steps is not None and step_index >= max_steps:
                break

            dt_step = self._derive_timestep(previous_time, snapshot_time)
            if self._kdt is not None and self._points is not None:
                sampled_velocity = self._sample_velocity_at_positions(field, positions)
            else:
                sampled_velocity = self._sample_velocity(field, num_particles)

            # Defensive: ensure sampled velocities are numeric. Some exported
            # velocity snapshots can contain NaNs/Infs (e.g., masked regions);
            # applying these to positions will quickly produce NaN trajectories.
            if not np.isfinite(sampled_velocity).all():
                # Attempt to salvage by replacing non-finite entries with zeros
                # (no advection for those particles this step) and warn once.
                bad_mask = ~np.isfinite(sampled_velocity)
                sampled_velocity = np.where(bad_mask, 0.0, sampled_velocity)
                print(
                    "Warning: encountered non-finite velocities in snapshot; "
                    "replacing them with zeros to avoid NaN trajectories."
                )

            diffusion_noise = self._diffusion_noise(num_particles, dt_step)
            if not np.isfinite(diffusion_noise).all():
                # If diffusion computation somehow produced non-finite values,
                # replace with zeros to avoid corrupting positions.
                diffusion_noise = np.where(~np.isfinite(diffusion_noise), 0.0, diffusion_noise)

            positions = positions + sampled_velocity * dt_step + diffusion_noise

            history.append(positions.copy())
            if self.respect_snapshot_timing:
                timesteps.append(snapshot_time)
                previous_time = snapshot_time
            else:
                previous_time = previous_time + dt_step if previous_time is not None else dt_step
                timesteps.append(previous_time)

        history_array = np.stack(history, axis=0)
        return TrajectoryResult(history=history_array, timesteps=timesteps)

    def _derive_timestep(self, previous_time: float | None, snapshot_time: float) -> float:
        if self.respect_snapshot_timing and previous_time is not None:
            dt_step = snapshot_time - previous_time
            if dt_step > 0:
                return dt_step
        return self.dt

    def _initialize_particles(self, num_particles: int) -> np.ndarray:
        """Seed particles from a standard normal distribution."""
        return self.rng.normal(loc=0.0, scale=1.0, size=(num_particles, 2))

    def _sample_velocity(self, field: np.ndarray, num_particles: int) -> np.ndarray:
        array = np.asarray(field)
        if array.ndim != 2 or array.shape[1] < 2:
            raise ValueError("Velocity field must have shape (N, >=2)")
        planar = array[:, :2]
        indices = self.rng.integers(0, planar.shape[0], size=num_particles)
        return planar[indices]

    def _sample_velocity_at_positions(self, field: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Look up velocities at particle positions via nearest neighbour KDTree."""
        if self._kdt is None or self._points is None:
            return self._sample_velocity(field, positions.shape[0])

        planar = np.asarray(field)[:, :2]
        _, idx = self._kdt.query(positions)
        idx = np.clip(idx, 0, planar.shape[0] - 1)
        return planar[idx]

    def _diffusion_noise(self, num_particles: int, dt_step: float) -> np.ndarray:
        if self.diffusion <= 0:
            return np.zeros((num_particles, 2))
        scale = np.sqrt(2.0 * self.diffusion * dt_step)
        return self.rng.normal(loc=0.0, scale=scale, size=(num_particles, 2))
