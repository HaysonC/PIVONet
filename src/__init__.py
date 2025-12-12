"""PIVONet: Physics-Informed Variational ODE Networks.

===================================================================================
OVERVIEW
===================================================================================
PIVONet is a comprehensive Python framework that integrates Computational Fluid
Dynamics (CFD) simulations with Physics-Informed Variational ODE (PDE) networks
for hybrid modeling and trajectory prediction of thermal and flow systems.

===================================================================================
ARCHITECTURE
===================================================================================
The package is organized into multiple functional subsystems:

    src/
    ├── app/              [Deprecated: Legacy Streamlit GUI - not recommended]
    ├── cfd/              CFD ingestion, particle trajectory simulation, PyFR support
    ├── cli/              Conversational Rich CLI interface
    ├── interfaces/       Protocol definitions (config, data, trajectories, models)
    ├── modeling/         Neural network training infrastructure
    ├── networks/         ODE models (CNF, variational SDE, neural networks)
    ├── utils/            Shared helpers (paths, config, orchestrator)
    ├── visualization/    Trajectory plotting, velocity rendering, Taichi viewer
    └── workflows/        Standalone Python scripts for batch processing

===================================================================================
SYSTEM FLOW DIAGRAM
===================================================================================

    ┌─────────────────────────────────────────────────────────────────────┐
    │  User Input: CLI (pivo) - Conversational Interface                 │
    └──────────────────────┬──────────────────────────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────────────────────────┐
    │  1. Data Ingestion Layer                                           │
    │  ├─ Load CFD velocity snapshots via PyFR or .npy files            │
    │  ├─ Resolve simulation configuration from config.yml             │
    │  └─ interfaces/data_sources → NpyVelocityFieldSource              │
    └──────────────────────┬──────────────────────────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────────────────────────┐
    │  2. Particle Trajectory Simulation                                 │
    │  ├─ Integrate trajectories in velocity field (Langevin/ODE)       │
    │  ├─ cfd/particle_trajectories → ParticleTrajectorySimulator       │
    │  └─ Output: TrajectoryResult (positions, velocities, times)       │
    └──────────────────────┬──────────────────────────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────────────────────────┐
    │  3. Dataset Preparation                                            │
    │  ├─ modeling/datasets → CFDTrajectorySequenceDataset              │
    │  ├─ Normalize, batch trajectories for training                    │
    │  └─ PyTorch DataLoader integration                                │
    └──────────────────────┬──────────────────────────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────────────────────────┐
    │  4. Neural ODE Model Training                                      │
    │  ├─ networks/cnf → CNFModel (Continuous Normalizing Flows)        │
    │  ├─ networks/variational_sde → VariationalSDEModel                │
    │  ├─ modeling/trainers → Trainer class with checkpoint management  │
    │  └─ Output: Trained model checkpoint (.pt file)                   │
    └──────────────────────┬──────────────────────────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────────────────────────┐
    │  5. Inference & Visualization                                      │
    │  ├─ visualization/trajectories → TrajectoryPlotter                │
    │  ├─ visualization/velocity_field → VelocityFieldPlotter           │
    │  ├─ visualization/viewer_taichi → GPU-accelerated 3D viewer      │
    │  └─ workflows/render_velocity_animations → Velocity field GIFs    │
    └──────────────────────┬──────────────────────────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────────────────────────┐
    │  6. Experiment Orchestration                                       │
    │  ├─ YAML-driven pipeline execution (utils/orchestrator)           │
    │  ├─ workflows/*.py scripts invoked sequentially                   │
    │  └─ Progress tracking, result caching, error handling             │
    └──────────────────────────────────────────────────────────────────────┘

===================================================================================
KEY CONCEPTS
===================================================================================

**Trajectory**: A sequence of particle positions evolving through space-time
under velocity field or learned dynamics. Stored as numpy arrays or .npz bundles.

**Velocity Field Source**: Interface for loading CFD data (.npy or PyFR) with
uniform access to (x, y, z) → velocity components.

**Neural ODE**: Continuous dynamical system parameterized by neural network.
Two variants implemented:
  - CNF (Continuous Normalizing Flow): Learns flow via invertible transformations
  - Variational SDE: Probabilistic model with encoder/decoder

**LaunchOptions**: Dataclass capturing all CLI parameters (particles, dt,
max_steps, velocity source, model type, device). Used across all interfaces.

**TrajectoryResult**: Immutable bundle containing positions, velocities, times,
metadata. Can be serialized/deserialized from .npz files.

===================================================================================
CORE TYPES & STRUCTURES
===================================================================================

from src.interfaces import (
    SimulationConfig,          # Configuration loaded from config.yml
    VelocityFieldSource,       # Abstract interface for velocity data
    NpyVelocityFieldSource,    # NPY file implementation
    TrajectoryResult,          # Particles over time
    TrajectoryVisualizer,      # Protocol for ploting trajectories
    ModelingConfig,            # Neural network training hyperparameters
)

from src.networks import (
    CNFModel,                  # Continuous Normalizing Flow model
    VariationalSDEModel,       # Variational SDE model
    MLP,                       # Multilayer perceptron backbone
    Encoder,                   # Trajectory sequence encoder
)

from src.modeling import (
    CFDTrajectorySequenceDataset,  # PyTorch Dataset
    Trainer,                       # Training orchestration
)

===================================================================================
MEMORY OWNERSHIP
===================================================================================

numpy arrays: Owned by TrajectoryResult, returned by reference. Caller should
not modify to avoid side effects.

PyTorch tensors: Device placement (CPU/GPU/MPS) controlled via LaunchOptions.
Models own their weights. Gradients computed during training; zeroed after
backward pass in standard training loops.

CFD data: Lazy-loaded from disk only when accessed (NpyVelocityFieldSource).
Multiple particle simulations can reference same velocity source concurrently.

Config: Singleton-like, loaded once at startup. Assumed immutable thereafter.

===================================================================================
USAGE EXAMPLE: END-TO-END WORKFLOW
===================================================================================

# 1. Load configuration and data source
from src.utils.config import load_config
from src.interfaces.data_sources import NpyVelocityFieldSource
from src.cfd.particle_trajectories import ParticleTrajectorySimulator

config = load_config()
velocity_source = NpyVelocityFieldSource(config.velocity_dir)
simulator = ParticleTrajectorySimulator(velocity_source=velocity_source)

# 2. Simulate trajectories
result = simulator.simulate(
    n_particles=100,
    max_steps=50,
    dt=0.01,
)

# 3. Prepare dataset for training
from src.modeling.datasets import CFDTrajectorySequenceDataset
dataset = CFDTrajectorySequenceDataset(trajectory_bundles=[result])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

# 4. Create and train model
from src.networks.cnf import CNFModel
model = CNFModel(state_dim=2, hidden_dim=64)
trainer = Trainer(model, device="mps")
trainer.train(dataloader, num_epochs=100)

# 5. Visualize results
from src.visualization.trajectories import TrajectoryPlotter
plotter = TrajectoryPlotter()
plotter.plot_trajectories(result, save_path="output.png")

===================================================================================
CONSTRAINTS & ASSUMPTIONS
===================================================================================

1. Python version ≥ 3.10 (f-strings, PEP 604 unions, match statements).
2. All trajectories assumed to start from same initial condition or distributions.
3. Velocity fields assumed quasi-stationary or periodic (time-independent or
   periodic in time).
4. Neural ODE training assumes bounded trajectories (no extreme divergence).
5. GPU/MPS availability detected at runtime; falls back to CPU if unavailable.
6. Project structure assumes read/write access to data/ and cache/ directories.

===================================================================================
ENTRY POINTS
===================================================================================

Console script: pivo
    Launches conversational CLI defined in src.cli.main

Direct imports:
    from src.main import main  # Unified entry point (CLI launcher)

===================================================================================
"""

__version__ = "1.0.0"
__author__ = "Hei Shing Cheung, David Lin, Ethan Long"
__license__ = "MIT"

__all__ = [
    "__version__",
    "__author__",
    "__license__",
]
