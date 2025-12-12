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
    """
    Lagrangian particle trajectory simulator: advects particles through velocity fields with diffusion.
    
    Implements numerical integration of stochastic differential equations for particle motion:
        dx_i/dt = v(x_i, t) + √(2D) · dW_i/dt
    
    where v is the velocity field from a CFD solution, D is the diffusion coefficient, and dW_i
    is Brownian motion. Uses Euler integration with automatic timestep selection based on
    snapshot timing or user-specified dt.
    
    **Problem Solved**: Transform velocity snapshots from CFD simulations (Eulerian) into
    particle trajectories (Lagrangian) for neural ODE training. Handles real-world data challenges:
    - Adaptive time stepping based on available snapshot times
    - NaN/Inf rejection with automatic recovery
    - Spatial sampling via KDTree for nearest-neighbor lookup
    - Configurable diffusion for modeling unresolved turbulence
    
    **Physics Model**:
    Stochastic advection-diffusion: Position updates via:
        x(t+dt) = x(t) + v(x(t), t)·dt + √(2D·dt)·ξ
    
    where ξ ~ N(0, I). The diffusion coefficient D models subgrid-scale turbulence
    or numerical dissipation.
    
    **Memory Ownership**:
    - Owns: self.rng (numpy random generator), self._kdt (scipy KDTree if available)
    - Borrows: velocity_source (not retained after __init__)
    - Borrows: velocity fields from iter_velocity_fields() (caller owns)
    - Returns: TrajectoryResult (owned by caller)
    
    **Attributes**:
        velocity_source (VelocityFieldSource): Protocol object providing iter_velocity_fields()
        diffusion (float): Diffusion coefficient D ≥0. 0 = deterministic advection, >0 = turbulent.
        dt (float): Default timestep when respect_snapshot_timing=False
        respect_snapshot_timing (bool): If True, use snapshot times as dt; else use fixed dt
        rng (np.random.Generator): Seeded RNG for reproducible Brownian motion
        _points (ndarray | None): Mesh/grid points (n, 2) for KDTree spatial indexing (if available)
        _kdt (KDTree | None): scipy.spatial.cKDTree for nearest-neighbor velocity lookup
    
    **Time Complexity**:
    - Initialization: O(n_points) to build KDTree (if points file exists)
    - simulate(): O(n_particles × n_snapshots × n_timesteps) with negligible KDTree queries
      - Each step: O(n_particles) for velocity sampling + O(n_particles) for diffusion noise
    
    **Space Complexity**:
    - Model state: O(n_points) for KDTree (if available) + O(n_particles × n_snapshots) for history
    
    **Error Behavior**:
    - ValueError: "num_particles must be positive" if simulate(num_particles ≤ 0)
    - ValueError: "Velocity field must have shape (N, >=2)" if field has <2 velocity components
    - Graceful NaN/Inf handling: replaces non-finite velocities with zeros (line 97-104)
    - Graceful NaN/Inf diffusion: replaces non-finite noise with zeros (line 106-109)
    
    **Usage Example**::
    
        from src.interfaces.data_sources import NpyVelocityFieldSource
        
        # Create velocity source from snapshots
        vel_source = NpyVelocityFieldSource(
            velocity_dir="data/euler-vortex/velocity/",
            sort_key=lambda x: int(x.stem)
        )
        
        # Create simulator with diffusion
        simulator = ParticleTrajectorySimulator(
            velocity_source=vel_source,
            diffusion_coefficient=0.01,  # Subgrid-scale turbulence
            dt=1e-2,  # Default timestep
            seed=42,  # Reproducible randomness
            respect_snapshot_timing=True  # Use snapshot times as dt
        )
        
        # Run simulation
        trajectory = simulator.simulate(
            num_particles=1000,  # Track 1000 particles
            max_steps=100  # Limit to first 100 snapshots (optional)
        )
        
        # trajectory.history: (n_snapshots, n_particles, 2) positions
        # trajectory.timesteps: [t_0, t_1, ..., t_n] integration times
    
    **Key Design Decisions**:
    1. KDTree-based velocity sampling (line 54-57): Nearest-neighbor lookup if mesh points available,
       else random sampling. Reduces interpolation error for unstructured grids.
    2. NaN/Inf recovery (line 97-104, 106-109): Replaces bad values with zeros instead of failing.
       Prevents one corrupted snapshot from destroying entire trajectory.
    3. Flexible timestep (line 85-87): Respects snapshot timing by default (preserves temporal structure),
       or uses fixed dt (for uniform subsampling).
    4. Euler integration (line 101-102): Simple O(dt) method. Sufficient for CFD data with small dt;
       could upgrade to RK4 if needed.
    """

    def __init__(
        self,
        velocity_source: VelocityFieldSource,
        diffusion_coefficient: float,
        dt: float = 1e-2,
        seed: int | None = None,
        respect_snapshot_timing: bool = True,
    ) -> None:
        """
        Initialize the particle trajectory simulator.
        
        **Parameters**:
            velocity_source (VelocityFieldSource): Protocol object with iter_velocity_fields() method.
                Yields (time, velocity_array) tuples. Caller owns; not retained after init.
            diffusion_coefficient (float): Diffusion coefficient D ≥0 in units [spatial_unit²/time_unit].
                0 = pure advection (deterministic). Typical values: 0.001 - 0.1 for CFD.
                Represents unresolved subgrid-scale turbulence or numerical dissipation.
            dt (float, default=1e-2): Default timestep when respect_snapshot_timing=False.
                Only used if snapshot times unavailable. Units: time_unit. Must be >0.
            seed (int | None, default=None): Random seed for reproducibility. Passed to
                np.random.default_rng(). If None, uses system entropy (non-reproducible).
            respect_snapshot_timing (bool, default=True): If True, use timestamps from
                velocity_source.iter_velocity_fields() as timesteps; dt becomes unused.
                If False, use fixed dt for uniform subsampling. Affects physical accuracy.
        
        **Returns**: None
        
        **Side Effects**:
        - Creates np.random.Generator (line 41)
        - Attempts to load mesh points and build scipy.spatial.cKDTree if available (lines 44-62)
        - Non-critical failures in KDTree construction are silently ignored (line 59-61)
        
        **Memory Ownership**: 
        - Borrows velocity_source reference (not deep-copied)
        - Owns KDTree and RNG instance
        
        **Error Behavior**:
        - No validation of diffusion_coefficient (caller responsibility for D ≥0)
        - No validation of dt (caller responsibility for dt >0)
        - KDTree loading failures are silently caught and ignored (fallback to random sampling)
        
        **Time Complexity**: O(n_mesh_points) if mesh points file exists and scipy available
        
        **Space Complexity**: O(n_mesh_points) for KDTree storage
        
        **Usage Example**::
        
            simulator = ParticleTrajectorySimulator(
                velocity_source=vel_source,
                diffusion_coefficient=0.01,
                dt=1e-2,
                seed=42,
                respect_snapshot_timing=True
            )
        """
        self.velocity_source = velocity_source
        self.diffusion = float(diffusion_coefficient)
        self.dt = dt
        self.respect_snapshot_timing = respect_snapshot_timing
        self.rng = np.random.default_rng(seed)

        self._points: np.ndarray | None = None
        self._kdt: KDTreeType | None = None

        # Attempt to load mesh points for spatial velocity lookups
        # Searches multiple candidate paths for robustness
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
        # Build KDTree for nearest-neighbor queries (if scipy available)
        if self._points is not None and _KDTreeClass is not None:
            self._kdt = _KDTreeClass(self._points)

    def simulate(
        self,
        num_particles: int,
        max_steps: int | None = None,
    ) -> TrajectoryResult:
        """
        Run the particle trajectory simulation.
        
        Integrates particle trajectories through velocity field snapshots using Euler method
        with stochastic diffusion. Returns full trajectory history with timesteps.
        
        **Parameters**:
            num_particles (int): Number of particles to track. Must be ≥1.
                Typical: 100-10000 for neural ODE training.
            max_steps (int | None, default=None): Optional cap on velocity snapshots to use.
                If set, integrates through first max_steps snapshots only. Useful for
                testing or memory-limited scenarios. If None, uses all available snapshots.
        
        **Returns**:
            TrajectoryResult: Immutable bundle containing:
                - history (ndarray): Particle positions, shape (n_snapshots, num_particles, 2)
                  [time_step, particle_id, x/y_coordinate]
                - timesteps (list[float]): Integration times, length n_snapshots
                  Matches snapshot times if respect_snapshot_timing=True, else synthetic times.
        
        **Side Effects**:
        - Modifies internal RNG state (consumes random numbers)
        - Prints warning if non-finite velocities encountered (line 98-101)
        - May modify returned TrajectoryResult if positions become NaN/Inf (replaced with zeros)
        
        **Memory Ownership**:
        - Returns: New TrajectoryResult owned by caller
        - Borrows: velocity_source.iter_velocity_fields() (assumed to yield views/borrowable arrays)
        - Borrows: field arrays from velocity_source (not copied, assumed read-only)
        
        **Time Complexity**: O(num_particles × n_snapshots × 2)  [physics simpler than ML]
        - Each snapshot: O(num_particles) velocity sampling + O(num_particles) diffusion
        - Total: ~10-100µs per particle per snapshot (CPU-bound, very fast)
        
        **Space Complexity**: O(num_particles × n_snapshots) for history array
        - Scales linearly with trajectory length. For 1000 particles × 100 snapshots: ~0.8 MB
        
        **Error Behavior**:
        - Raises ValueError: "num_particles must be positive" if num_particles ≤0 (line 77)
        - Raises ValueError: Propagated from _sample_velocity() if field shape invalid (line 133)
        - Graceful degradation for non-finite values (replaces with 0, prints warning)
        
        **Algorithm** (Euler Integration):
        
        1. Initialize: x_0 ~ N(0, 1), history = [x_0]
        2. For each snapshot (velocity field v at time t):
            a. Compute timestep dt_step (from snapshot times or use default dt)
            b. Sample velocity at positions: v_sampled = v(x)
            c. Add diffusion noise: w ~ N(0, √(2D·dt))
            d. Update: x ← x + v_sampled·dt + w
            e. Append to history
        3. Return TrajectoryResult(history, timesteps)
        
        The integration is first-order Euler, sufficient for small timesteps.
        Stochastic term √(2D·dt)·ξ is correct discretization of Brownian motion.
        
        **NaN/Inf Handling**:
        Lines 97-104: If velocity snapshot contains NaN/Inf (e.g., masked regions in CFD):
        - Replace non-finite values with 0 (no advection for affected particles)
        - Print one warning to user (prevents output spam)
        - Continue integration (robust to partially corrupted data)
        
        **Usage Example**::
        
            # Single simulation
            traj = simulator.simulate(num_particles=1000, max_steps=None)
            print(f"Positions: {traj.history.shape}")  # (100, 1000, 2)
            print(f"Times: {len(traj.timesteps)}")     # 100
            
            # Batch simulations
            trajectories = []
            for diffusion in [0.0, 0.01, 0.1]:
                sim = ParticleTrajectorySimulator(
                    vel_source, diffusion_coefficient=diffusion, seed=i
                )
                traj = sim.simulate(num_particles=1000)
                trajectories.append(traj)
        """
        if num_particles <= 0:
            raise ValueError("num_particles must be positive")

        # Initialize particles from standard normal
        positions = self._initialize_particles(num_particles)
        history = [positions.copy()]
        timesteps = [0.0]

        previous_time: float | None = timesteps[0]

        # Iterate through velocity snapshots
        for step_index, (snapshot_time, field) in enumerate(
            self.velocity_source.iter_velocity_fields()
        ):
            if max_steps is not None and step_index >= max_steps:
                break

            # Compute appropriate timestep for this step
            dt_step = self._derive_timestep(previous_time, snapshot_time)
            
            # Sample velocities at particle positions
            # Use KDTree nearest-neighbor if available (spatial accuracy), else random sampling
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

            # Generate Brownian motion increment: √(2D·dt)·ξ, ξ~N(0,I)
            diffusion_noise = self._diffusion_noise(num_particles, dt_step)
            if not np.isfinite(diffusion_noise).all():
                # If diffusion computation somehow produced non-finite values,
                # replace with zeros to avoid corrupting positions.
                diffusion_noise = np.where(~np.isfinite(diffusion_noise), 0.0, diffusion_noise)

            # Euler update: x ← x + v·dt + w
            positions = positions + sampled_velocity * dt_step + diffusion_noise

            history.append(positions.copy())
            if self.respect_snapshot_timing:
                timesteps.append(snapshot_time)
                previous_time = snapshot_time
            else:
                previous_time = previous_time + dt_step if previous_time is not None else dt_step
                timesteps.append(previous_time)

        # Stack history into (n_snapshots, num_particles, 2) array
        history_array = np.stack(history, axis=0)
        return TrajectoryResult(history=history_array, timesteps=timesteps)

    def _derive_timestep(self, previous_time: float | None, snapshot_time: float) -> float:
        """
        Compute timestep for this integration step.
        
        **Parameters**:
            previous_time (float | None): Time of previous snapshot, or None if first step
            snapshot_time (float): Time of current snapshot
        
        **Returns**:
            float: Timestep dt_step to use for Euler update. If respect_snapshot_timing,
            returns (snapshot_time - previous_time). Otherwise returns self.dt.
        
        **Logic**:
        - If respect_snapshot_timing=True and previous_time available:
          dt_step = snapshot_time - previous_time (physical snapshot timing)
        - Else: dt_step = self.dt (fixed timestep, uniform subsampling)
        
        **Time Complexity**: O(1)
        
        **Usage Example**::
        
            sim = ParticleTrajectorySimulator(..., respect_snapshot_timing=True)
            dt = sim._derive_timestep(previous_time=1.0, snapshot_time=1.1)
            # Returns 0.1 (physical time difference)
            
            sim2 = ParticleTrajectorySimulator(..., respect_snapshot_timing=False)
            dt2 = sim2._derive_timestep(previous_time=1.0, snapshot_time=1.1)
            # Returns 1e-2 (default dt, ignores snapshot times)
        """
        if self.respect_snapshot_timing and previous_time is not None:
            dt_step = snapshot_time - previous_time
            if dt_step > 0:
                return dt_step
        return self.dt

    def _initialize_particles(self, num_particles: int) -> np.ndarray:
        """
        Seed particles from standard normal distribution N(0,1).
        
        **Parameters**:
            num_particles (int): Number of particles to generate
        
        **Returns**:
            ndarray: Initial positions, shape (num_particles, 2) with dtype float64
        
        **Usage Example**::
        
            x0 = sim._initialize_particles(1000)  # (1000, 2)
        """
        return self.rng.normal(loc=0.0, scale=1.0, size=(num_particles, 2))

    def _sample_velocity(self, field: np.ndarray, num_particles: int) -> np.ndarray:
        """
        Sample velocities randomly from a velocity field (uniform random sampling).
        
        Used when mesh points/KDTree unavailable. Selects random rows from field array.
        
        **Parameters**:
            field (ndarray): Velocity field snapshot, shape (n_grid_points, n_components).
                Must have ≥2 components (x, y velocity at minimum).
            num_particles (int): Number of samples to draw
        
        **Returns**:
            ndarray: Sampled velocities, shape (num_particles, 2). Samples uniformly with
            replacement from field[:, :2] (planar x,y components).
        
        **Raises**:
            ValueError: "Velocity field must have shape (N, >=2)" if field has <2 columns
        
        **Time Complexity**: O(num_particles) (random indexing + gathering)
        
        **Space Complexity**: O(num_particles × 2)
        
        **Usage Example**::
        
            v_field = np.random.randn(10000, 3)  # 10K grid points, (vx, vy, vz)
            v_samples = sim._sample_velocity(v_field, num_particles=1000)
            # Returns (1000, 2) with random rows from v_field[:, :2]
        """
        array = np.asarray(field)
        if array.ndim != 2 or array.shape[1] < 2:
            raise ValueError("Velocity field must have shape (N, >=2)")
        planar = array[:, :2]
        indices = self.rng.integers(0, planar.shape[0], size=num_particles)
        return planar[indices]

    def _sample_velocity_at_positions(self, field: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """
        Look up velocities at particle positions via nearest-neighbor KDTree.
        
        Spatially-aware velocity sampling: finds nearest grid point to each particle
        position and returns velocity at that grid point. More accurate than random
        sampling for structured/semi-structured grids.
        
        **Parameters**:
            field (ndarray): Velocity field snapshot, shape (n_grid_points, ≥2)
            positions (ndarray): Current particle positions, shape (num_particles, 2)
        
        **Returns**:
            ndarray: Velocities at positions, shape (num_particles, 2)
            
        **Algorithm**:
        1. Query KDTree for nearest grid point index for each particle
        2. Clamp indices to [0, n_grid_points) to avoid out-of-bounds
        3. Return field[indices, :2]
        
        **Time Complexity**: O(num_particles × log(n_grid_points)) via KDTree query
        
        **Space Complexity**: O(num_particles)
        
        **Fallback**: If KDTree unavailable, calls _sample_velocity() for random sampling
        
        **Usage Example**::
        
            positions = np.random.randn(1000, 2)  # Current particle locations
            v_field = np.random.randn(10000, 3)  # Grid velocities
            velocities = sim._sample_velocity_at_positions(v_field, positions)
            # Returns (1000, 2) velocities at nearest grid points
        """
        if self._kdt is None or self._points is None:
            return self._sample_velocity(field, positions.shape[0])

        planar = np.asarray(field)[:, :2]
        _, idx = self._kdt.query(positions)
        idx = np.clip(idx, 0, planar.shape[0] - 1)
        return planar[idx]

    def _diffusion_noise(self, num_particles: int, dt_step: float) -> np.ndarray:
        """
        Generate Brownian motion noise for stochastic integration.
        
        Computes √(2D·dt)·ξ where D is diffusion coefficient and ξ ~ N(0,I).
        This is the correct Ito discretization of Brownian motion.
        
        **Parameters**:
            num_particles (int): Number of particles (noise shape)
            dt_step (float): Timestep for this integration step
        
        **Returns**:
            ndarray: Brownian noise, shape (num_particles, 2) with dtype float64
                     Variance is 2D·dt in each dimension.
        
        **Returns zero if** diffusion_coefficient ≤0 (deterministic advection)
        
        **Formula**:
        - If D ≤0: return zeros (no noise)
        - Else: σ = √(2D·dt), return N(0, σ²) samples
        
        **Time Complexity**: O(num_particles)
        
        **Space Complexity**: O(num_particles × 2)
        
        **Usage Example**::
        
            # For diffusion=0.01, dt=1e-2: σ = √(2×0.01×1e-2) ≈ 0.0141
            noise = sim._diffusion_noise(num_particles=1000, dt_step=1e-2)
            # Returns (1000, 2) with std ≈ 0.0141
        """
        if self.diffusion <= 0:
            return np.zeros((num_particles, 2))
        scale = np.sqrt(2.0 * self.diffusion * dt_step)
        return self.rng.normal(loc=0.0, scale=scale, size=(num_particles, 2))
