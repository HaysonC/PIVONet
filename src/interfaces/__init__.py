"""Shared interfaces and protocols for the ODE/CFD toolchain.

===================================================================================
OVERVIEW
===================================================================================
This package defines the contract between different parts of PIVONet via:
  - Abstract base classes (Protocol)
  - Dataclasses for configuration and data
  - Type hints for consistency

Design pattern: Interface segregation. Each submodule implements exactly
the protocol it needs, allowing loose coupling and easy testing/mocking.

===================================================================================
SUBMODULE STRUCTURE
===================================================================================

config.py:
    SimulationConfig - Immutable configuration dataclass
    Fields:
      - version: str
      - diffusion_constant: float
      - data_root: Path
      - velocity_subdir: str
      - trajectory_particles: int
      - trajectory_steps: int | None
      - trajectory_dt: float

data_sources.py:
    VelocityFieldSource (Protocol) - Abstract interface
      - Methods: query(x, y[, z], t) → velocity vector
    NpyVelocityFieldSource - Implementation via .npy files
      - Lazy loads snapshots on demand
      - Supports spatial interpolation (RectBivariateSpline)

trajectories.py:
    TrajectoryResult - Immutable trajectory data bundle
    Attributes:
      - positions: ndarray(n_particles, n_steps, spatial_dim)
      - velocities: ndarray(n_particles, n_steps, spatial_dim)
      - times: ndarray(n_steps,)
      - metadata: dict (source, n_particles, dt, max_step)
    Methods:
      - to_numpy() → dict of arrays for saving
      - from_numpy() → reconstruct from saved dict

launch_options.py:
    LaunchOptions - Frozen dataclass for CLI/GUI parameters
    Fields:
      - velocity_dir: Path
      - particles: int
      - max_steps: int
      - dt: float
      - model_type: str
      - device: str
      - output_dir: Path

modeling.py:
    ModelingConfig - Neural network hyperparameters
    Fields:
      - hidden_dim: int
      - depth: int
      - lr: float
      - batch_size: int
      - epochs: int
      - device: str

visualization.py:
    TrajectoryVisualizer (Protocol) - Abstract plotter
      - Methods: plot(result) → Figure
    PlotArtifact - Result metadata for saved plots

cfd.py:
    CFDSimulator (Protocol) - Abstract CFD runner
    SimulationResult - Velocity and mesh data
    PyFRBackend - PyFR-specific implementation

pyfr.py:
    PyFRSimulationConfig - PyFR-specific parameters
    PyFRBackend - Wrapper for PyFR CLI

===================================================================================
DESIGN PRINCIPLES
===================================================================================

1. Immutability: Config dataclasses are frozen (hashable, thread-safe)
2. Type safety: All interfaces specify expected types
3. Serialization: All dataclasses implement .to_dict()/.from_dict()
4. Error messages: Clear, actionable error text for common mistakes
5. Extensibility: New data sources can implement VelocityFieldSource

Example: Adding custom velocity source

    from src.interfaces.data_sources import VelocityFieldSource
    import numpy as np
    
    class CustomVelocitySource(VelocityFieldSource):
        def query(self, x, y, z=None, t=0.0):
            \"\"\"Return velocity at (x, y[, z]) at time t.\"\"\"
            # Example: analytical vortex
            r = np.sqrt(x**2 + y**2)
            v_theta = np.sin(r) / (r + 1e-6)
            return np.stack([-y * v_theta, x * v_theta], axis=-1)

===================================================================================
DATA STRUCTURES DIAGRAM
===================================================================================

    SimulationConfig (from config.yml)
            │
            ▼
    VelocityFieldSource (abstract)
            │
            ├─ NpyVelocityFieldSource
            ├─ AnalyticalVelocitySource
            └─ PyFRBackend
            │
            ▼
    TrajectorySimulator.simulate()
            │
            ▼
    TrajectoryResult
        ├─ positions
        ├─ velocities
        ├─ times
        └─ metadata
            │
            ├─ Training
            │   └─ DataLoader
            │       ├─ ModelingConfig
            │       └─ Model (CNF/SDE)
            │
            └─ Visualization
                └─ TrajectoryVisualizer
                    └─ PlotArtifact

===================================================================================
MEMORY MANAGEMENT
===================================================================================

SimulationConfig: Singleton-like, typically loaded once. Immutable so safe
to share across threads. ~1KB memory.

TrajectoryResult: Owns numpy arrays (dense storage). Size ~ 1-100 MB for
typical simulations (100-1000 particles, 50-1000 steps, 2-3D).

VelocityFieldSource: Lazy loading via NpyVelocityFieldSource.  Velocity
snapshots loaded from disk on first query, then cached. Typical CFD snapshot:
1-10 MB per timestep.

===================================================================================
TYPE SAFETY & VALIDATION
===================================================================================

All dataclasses include __post_init__ validation:
  - Positive dimensions
  - Valid enum values
  - Sensible default ranges

Example:

    @dataclass(frozen=True)
    class ModelingConfig:
        hidden_dim: int = 64
        depth: int = 3
        
        def __post_init__(self):
            if self.hidden_dim < 1:
                raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
            if self.depth < 1:
                raise ValueError(f"depth must be positive, got {self.depth}")

===================================================================================
"""

# ./src/interfaces/__init__.py
"""Shared interfaces for the ODE/CFD toolchain."""

from .config import SimulationConfig
from .data_sources import NpyVelocityFieldSource, VelocityFieldSource
from .trajectories import TrajectoryResult
from .visualization import PlotArtifact, TrajectoryVisualizer
from .modeling import ModelingConfig

__all__ = [
	"SimulationConfig",
	"VelocityFieldSource",
	"NpyVelocityFieldSource",
	"TrajectoryResult",
	"TrajectoryVisualizer",
	"PlotArtifact",
	"ModelingConfig",
]
# This directory contains interface definitions for various components 
# It controls how top level parts for each module interact and understand each other