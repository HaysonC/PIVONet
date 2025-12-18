"""CFD (Computational Fluid Dynamics) simulation and trajectory integration module.

===================================================================================
OVERVIEW
===================================================================================
This package provides utilities for loading CFD velocity data, simulating particle
trajectories through velocity fields, and managing PyFR-based simulations.

Main responsibilities:
  - Load velocity snapshots from disk (numpy arrays)
  - Integrate particle trajectories using ODE solvers
  - Export trajectory data as serialized bundles
  - Interface with PyFR for running full CFD simulations

===================================================================================
SUBMODULE STRUCTURE
===================================================================================

particle_trajectories:
    ParticleTrajectorySimulator - Main class for trajectory integration
    Supports Euler, RK4 and scipy ODE solvers
    Langevin dynamics with optional diffusion

evaluation:
    Metrics for trajectory quality assessment
    Path length, mean squared displacement, overlap statistics

modeling:
    Preprocessing of CFD outputs
    Normalization, field interpolation

pipeline:
    Orchestration utilities
    Wraps multiple simulation steps

pyfr_simulation:
    PyFR integration layer
    Setup, execution, output parsing

visualization:
    Postprocessing utilities for velocity fields
    Frame generation for animations

cfd.py:
    Core interfaces for CFD (abstract base classes)

===================================================================================
KEY TYPES
===================================================================================

ParticleTrajectorySimulator:
    Attributes:
      - velocity_source: VelocityFieldSource
      - solver: str ('euler', 'rk4', 'scipy')
      - diffusion_coeff: float
    Methods:
      - simulate(n_particles, max_steps, dt) -> TrajectoryResult
      - _step(positions, velocities, dt) -> (new_pos, new_vel)

TrajectoryResult (from interfaces):
    Attributes:
      - positions: ndarray(n_particles, n_steps, spatial_dim)
      - velocities: ndarray(n_particles, n_steps, spatial_dim)
      - times: ndarray(n_steps,)
      - metadata: dict (particle_count, time_range, source)

===================================================================================
DATA FLOW DIAGRAM
===================================================================================

    VelocityFieldSource (interface)
            │
            ├─ NpyVelocityFieldSource → Load .npy files
            │
            ▼
    ParticleTrajectorySimulator
            │
    ┌───────┼───────┐
    │       │       │
    ▼       ▼       ▼
  Initial  Velocity ODE Solver
  Positions Query   (Euler/RK4/Scipy)
    │       │       │
    └───────┼───────┘
            │
            ▼
    TrajectoryResult
            │
    ┌───────┴───────┐
    │               │
    ▼               ▼
Metrics         Visualization
(evaluation)     (trajectories)

===================================================================================
USAGE PATTERNS
===================================================================================

# Basic trajectory simulation
from src.cfd.particle_trajectories import ParticleTrajectorySimulator
from src.interfaces.data_sources import NpyVelocityFieldSource

velocity_source = NpyVelocityFieldSource("data/2d-euler-vortex/velocity")
simulator = ParticleTrajectorySimulator(velocity_source)

result = simulator.simulate(
    n_particles=100,
    max_steps=50,
    dt=0.01,
)

# Save trajectory bundle
from src.utils.trajectory_io import save_trajectory_bundle
output_path = save_trajectory_bundle(result, "output/traj.npz")

# Evaluate trajectory quality
from src.cfd.evaluation import compute_path_length, compute_msd
path_lengths = compute_path_length(result.positions)
msd = compute_msd(result.positions)

===================================================================================
MEMORY MANAGEMENT
===================================================================================

Particle positions and velocities are stored as dense numpy arrays in-memory.
For large simulations (>10k particles, >1000 steps), memory usage scales as
O(n_particles × n_steps × spatial_dim).

Velocity field data is typically lazy-loaded from disk on demand, avoiding
full duplication when multiple simulations use the same CFD snapshot.

TrajectoryResult is immutable after creation; modifications should create new
instances to avoid side effects in downstream processing.

===================================================================================
CONSTRAINTS & ASSUMPTIONS
===================================================================================

1. Velocity fields assumed to be stationary or periodic in time
2. Particle trajectories must remain in domain (no boundary wrapping)
3. Spatial interpolation assumes regular grid or provided mesh_points
4. ODE solver must be stable for given dt (auto-select or validate externally)
5. Initial particle positions should avoid singularities in velocity field

===================================================================================
ERROR HANDLING
===================================================================================

ParticleTrajectorySimulator raises:
  - ValueError: if n_particles < 1 or max_steps < 1 or dt <= 0
  - FileNotFoundError: if velocity data files missing
  - RuntimeError: if solver diverges or NaN detected

Recommended patterns:
  try:
      result = simulator.simulate(...)
  except ValueError as e:
      logger.error(f"Invalid simulation parameters: {e}")
  except RuntimeError as e:
      logger.error(f"Simulation failed to converge: {e}")

===================================================================================
"""

from .modeling import HybridModel
from .pipeline import CFDPipeline
from .visualization import CFDVisualizer

__all__ = ["CFDPipeline", "CFDVisualizer", "HybridModel"]
