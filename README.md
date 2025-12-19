# PIVONet: Physics-Informed Variational ODE Networks

![GitHub Last Commit](https://img.shields.io/github/last-commit/HaysonC/ODE_CNF_thermal_motion?style=flat-square)
![PyTorch 2.9.1](https://img.shields.io/badge/torch-2.9.1-red?style=flat-square)
![Python 3.10+](https://img.shields.io/badge/python-3.11%2B-377ded?style=flat-square)
![License](https://img.shields.io/github/license/HaysonC/ODE_CNF_thermal_motion?style=flat-square)

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Usage Modes](#usage-modes)
6. [Core Algorithms](#core-algorithms)
7. [Project Structure](#project-structure)
8. [Advanced Examples](#advanced-examples)
9. [Troubleshooting](#troubleshooting)
10. [Contributing](#contributing)
11. [License](#license)

---

## Overview

**PIVONet** is a Physics-Informed Variational ODE (Ordinary Differential Equation) toolkit that integrates Computational Fluid Dynamics (CFD) simulations with neural ODE models for hybrid trajectory prediction and analysis of thermal and flow systems.

### Problem

Classical CFD solvers are computationally expensive (hours to days for complex geometries). Machine learning models often ignore physical constraints, leading to poor generalization. PIVONet bridges this gap by:

1. **Ingesting CFD velocity fields** from PyFR simulations or pre-computed snapshots
2. **Simulating particle trajectories** through these fields using ODE integration
3. **Training neural ODE models** (CNF + Variational SDE) to learn dynamics with uncertainty quantification
4. **Providing a conversational CLI interface** for all operations

### Key Features

- **Conversational CLI**: Interactive command-line interface (`pivo`) for all operations
- **Physics-Informed**: Preserves ODE structure; invertible transformations via CNF
- **Efficient**: Adjoint-method training reduces memory from O(steps) to O(1)
- **Flexible**: Works with any velocity field source (PyFR, analytical, pre-computed)
- **Reproducible**: YAML-driven experiment pipelines with checkpoint management

> [!WARNING]
> The Streamlit GUI is experimental and not recommended.
> When you run `pivo`, you will be prompted to choose **CLI** or **GUI** — choose **CLI**.

## Quick Start

### Installation

Requires **Python ≥ 3.10**. Clone the repository and install:

```bash
git clone https://github.com/HaysonC/ODE_CNF_thermal_motion
cd ODE_CNF_thermal_motion
pip install -r requirements.txt
pip install -e .
```

### Your First Simulation (2 minutes)

```bash
# Launch interactive CLI
pivo
```

> [!WARNING]
> You will be prompted to choose **CLI** or **GUI**.
> The GUI is experimental — choose **CLI**.

> [!NOTE]
> Pretrained checkpoints and example data are available here:
> https://drive.google.com/drive/folders/13ykGleAmZTNAz1lhR0x6FekYqyGYbper?usp=sharing

In the conversational CLI:

1. Choose `import` → **Model checkpoints (pretrained)** and select the downloaded folder. It would be under pretrained in the drive.
2. Choose `experiment` and select the dataset you imported checkpoints for (e.g. `2d-euler-vortex`).
3. When a training step detects an existing checkpoint, choose **Skip step (reuse cached checkpoint)**.
   - This runs inference using the imported pretrained models.

> [!TIP]
> Use `pivo --cli` to skip the interface prompt.

Expected output:

- Console-guided experiment run (no flags needed)
- Checkpoints detected under `cache/checkpoints/` and reused when you choose **Skip step**
- Inference outputs and plots written under `cache/artifacts/` (varies by experiment)

### Quick Tips

> **⚠️ Notice:** Default `limit` values in the experiment YAML files under `src/experiments/` have been reduced to lower particle/sample counts for faster, lightweight testing. Use the `--limit` or `--particles` CLI flags, or edit the YAML directly, to restore larger-scale runs when needed.

- **Reduce particle count for fast runs ( quick testing):**
    - Use the `--particles` flag when running the import workflow to override the config default. Example:

        ```bash
        python -m src.workflows.import_trajectories --particles 64 --output-dir data/demo
        ```

    - When training, use the `--limit` flag to cap the number of samples consumed for faster iterations:

        ```bash
        python -m src.workflows.train_cnf --bundles data/demo --limit 1000 --epochs 5
        ```

- **Pretrained checkpoints:**
    - Pretrained checkpoints are included in the repository cache under `cache/checkpoints/` (e.g. `cache/checkpoints/2d-euler-vortex_cnf/cnf_latest.pt`).
    - To reuse an existing checkpoint during training the CLI will detect and offer to reuse it. Programmatically you can load a checkpoint with:

        ```python
        from pathlib import Path
        from src.networks.cnf import CNFModel
        model = CNFModel(dim=2, hidden_dim=128)
        model.load_state_dict(torch.load(Path('cache/checkpoints/2d-euler-vortex_cnf/cnf_latest.pt')))
        model.eval()
        ```

    - To point the workflow at a checkpoint directory use `--ckpt-dir`:

        ```bash
        python -m src.workflows.train_cnf --ckpt-dir cache/checkpoints/2d-euler-vortex_cnf
        ```

    - If you want a single-file example for inference (no training), load the checkpoint into the model and run your inference script as above.

---

## Architecture

### System Overview

```
┌────────────────────────────────────────────────────────────────┐
│                       User Input (CLI)                         │
└──────────────────────────┬─────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│ 1. Data Ingestion Layer                                         │
│   • Load velocity snapshots (.npy, PyFR, or analytical)         │
│   • Parse config.yml for simulation parameters                  │
│   • Resolve project paths and cache directories                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│ 2. Particle Trajectory Simulation                               │
│   • Integrate ODE: dx/dt = V(x, t)  [Langevin + diffusion]      │
│   • Solvers: Euler, RK4, or scipy adaptive                      │
│   • Output: TrajectoryResult (positions, velocities, times)     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│ 3. Dataset Preparation                                          │
│   • Create PyTorch datasets with sliding windows                │
│   • Normalize and batch trajectories                            │
│   • Prepare DataLoader for training                             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│ 4. Neural ODE Model Training                                    │
│   • CNF (Continuous Normalizing Flows) or                       │
│   • Variational SDE (probabilistic dynamics)                    │
│   • Adjoint method for gradient computation                     │
│   • Checkpoint best model and save metrics                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│ 5. Inference & Visualization                                    │
│   • Generate trajectory animations (GIF/MP4)                    │
│   • Plot loss curves and trajectory metrics                     │
│   • Export results for downstream analysis                      │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow Diagram

```
VelocityField
    ├─ NpyVelocityFieldSource (disk-backed lazy loading)
    ├─ AnalyticalVelocitySource (computed on-the-fly)
    └─ PyFRBackend (via PyFR executable)
        │
        ▼
ParticleTrajectorySimulator
    • ODE solver (Euler, RK4, Adaptive)
    • Optional Langevin diffusion
    • Handles boundary conditions
        │
        ▼
TrajectoryResult (immutable)
    • positions: shape (n_particles, n_steps, spatial_dim)
    • velocities: shape (n_particles, n_steps, spatial_dim)
    • times: shape (n_steps,)
    • metadata: {source, n_particles, dt, max_step}
        │
        ├─ Save/Load via trajectory_io.py
        ├─ Dataset preparation (modeling/datasets.py)
        └─ Visualization (visualization/)
                │
                ├─ Static plots (matplotlib)
                ├─ Animations (matplotlib animation)
            └─ (Optional) interactive viewer (deprecated)
```

---

## Installation

### Requirements

- **Python 3.10+** (uses PEP 604 unions, match statements)
- **pip** or **conda**
- **CUDA 11.8+** or **Metal** (for GPU, optional but recommended)

### Full Installation (with optional dependencies)

```bash
# Clone repository
git clone https://github.com/HaysonC/ODE_CNF_thermal_motion
cd ODE_CNF_thermal_motion

# Install core dependencies
pip install -r requirements.txt

# Install PIVONet in editable mode
pip install -e .

# Optional: For PyFR CFD simulations
pip install pyfr[backend]  # Backend = 'cuda', 'openmp', 'opencl'

# Optional: For Jupyter notebooks
pip install jupyter jupyterlab
```

### Verify Installation

```bash
# Check console script is available
which pivo

# Test import
python -c "from src.main import main; print('✓ Import successful')"

# Run test
pivo --help
```

---

## Usage Modes

> [!WARNING]
> The GUI mode is unstable/experimental. Use the CLI.

### 1. Conversational CLI

Start the interactive CLI:

```bash
pivo
```

Menu options:

- **Import**: Choose **Flow data** (copy a dataset folder with `velocity/` into `data/<flow-name>/`) or **Model checkpoints (pretrained)**
- **Visualize**: Generate plots for trajectory bundles
- **Model**: Train encoder + CNF on trajectories
- **Run Experiments**: Execute YAML-defined pipelines

### 2. Programmatic API

Use PIVONet from Python:

```python
from src.cfd.particle_trajectories import ParticleTrajectorySimulator
from src.interfaces.data_sources import NpyVelocityFieldSource
from src.utils.trajectory_io import save_trajectory_bundle
from src.networks.cnf import CNFModel

# Simulate trajectories
velocity_source = NpyVelocityFieldSource("data/2d-euler-vortex/velocity")
simulator = ParticleTrajectorySimulator(velocity_source=velocity_source)
result = simulator.simulate(n_particles=100, max_steps=50, dt=0.01)
save_trajectory_bundle(result, "output/traj.npz")

# Train neural ODE
model = CNFModel(dim=2, hidden_dim=128)
# ... (training loop with DataLoader)
```

### 4. Command-Line Workflows

Run individual workflows directly:

```bash
# Simulate trajectories
python -m src.workflows.import_trajectories \
    --velocity-dir data/2d-euler-vortex/velocity \
    --particles 500 \
    --output-dir data/demo

# Train CNF model
python -m src.workflows.train_cnf \
    --trajectory data/demo/trajectories.npz \
    --epochs 100 \
    --output-dir cache/checkpoints/cnf_run1
```

> [!TIP]
> Pretrained inference workflow:
>
> 1) Download checkpoints/data from:
>    https://drive.google.com/drive/folders/13ykGleAmZTNAz1lhR0x6FekYqyGYbper?usp=sharing
> 2) Run `pivo` → `import` → `Model checkpoints (pretrained)` and import the folder.
> 3) Run `pivo` → `experiment` and select the corresponding dataset.
> 4) When prompted during training steps, choose **Skip step (reuse cached checkpoint)** to jump straight to inference.

### 5. YAML Experiment Pipelines

Define experiments in YAML:

```yaml
# src/experiments/my_experiment.yaml
name: "My Custom Experiment"
slug: "my-experiment"
description: "Simulate, train, and evaluate"

steps:
  - name: "Simulate Trajectories"
    script: "src/workflows/import_trajectories.py"
    params:
      velocity_dir: "data/2d-euler-vortex/velocity"
      particles: 200
      max_steps: 100

  - name: "Train CNF"
    script: "src/workflows/train_cnf.py"
    params:
      trajectory: "data/demo/trajectories.npz"
      epochs: 50
```

Run it:

```bash
## Recommended: run via the conversational CLI
pivo

# Then choose: experiment → my-experiment
```

---

## Core Algorithms

### 1. Continuous Normalizing Flows (CNF)

**Goal**: Learn invertible transformation of trajectories with tractable likelihood.

**Math**:

- Transform data via ODE: $\frac{dz}{dt} = f_\theta(z, t)$
- Log-probability change: $\frac{d(\log p)}{dt} = -\text{tr}\left(\frac{\partial f}{\partial z}\right)$
- Trace computed via Hutchinson estimator: $\text{tr}(J) \approx \epsilon^T \cdot \nabla_z(f \cdot \epsilon)$, where $\epsilon \sim N(0, I)$
- Loss: $L = -\log p(z_1)$ (averaged over batch)

**Implementation** (`src/networks/cnf.py`):

- `ODEFunc`: Neural network parameterizing $f_\theta$
- `CNFModel`: Wrapper with integration via `torchdiffeq.odeint_adjoint`
- Adjoint method reduces memory from $O(n_\text{steps})$ to $O(1)$

**Advantages**:

- Exact likelihood computation
- Invertible (can reverse trajectories)
- Memory efficient
- Theoretically grounded

**Limitations**:

- Deterministic (doesn't capture stochasticity)
- Sensitive to ODE solver hyperparameters

### 2. Variational SDE

**Goal**: Learn stochastic dynamics with explicit uncertainty.

**Math**:

- Encoder: $q(z|x) = N(\mu_\phi(x), \sigma^2_\phi(x))$
- Decoder: $x' = g_\psi(z, t)$
- Loss: $L = D_\text{KL}(q(z|x) \| p(z)) + \|x - x'\|^2$
- Reparameterization: $z = \mu + \sigma \cdot \epsilon$, $\epsilon \sim N(0, I)$

**Implementation** (`src/networks/variational_sde.py`):

- Encoder reduces trajectory to latent code
- Decoder maps latent code → predicted trajectory
- Beta-annealing schedule for KL weight

**Advantages**:

- Captures aleatoric uncertainty
- Handles stochastic dynamics
- Robust to outliers via probabilistic formulation

**Limitations**:

- Assumes Gaussian posteriors (may be restrictive)
- Larger model size than CNF

### 3. ODE Solvers

Three solver options for trajectory integration:


| Solver             | Accuracy  | Speed    | Stability           |
| ------------------ | --------- | -------- | ------------------- |
| **Euler**          | O(h)      | Fastest  | Low (CFL condition) |
| **RK4**            | O(h^4)    | Moderate | High                |
| **Scipy Adaptive** | Automatic | Slowest  | Highest             |

Selection: Use RK4 by default (good balance); switch to Euler for speed, Adaptive for stiff systems.

---

## Project Structure

```
ODE_CNF_thermal_motion/
├── README.md                                 # This file
├── LICENSE                                   # MIT License
├── config.yml                                # Global configuration
├── requirements.txt                          # Dependency pins
├── pyproject.toml                            # Package metadata
├── main.sh                                   # Convenience entry script
│
├── src/                                      # Core Python package
│   ├── __init__.py                           # Package-level documentation
│   ├── main.py                               # Unified entry point (CLI launcher)
│   │
│   ├── app/                                  # [Deprecated: Not recommended for use]
│   │   ├── __init__.py                       # [Legacy Streamlit UI]
│   │   └── ui.py                             # [Legacy Streamlit application]
│   │
│   ├── cfd/                                  # CFD simulations & utilities
│   │   ├── __init__.py
│   │   ├── particle_trajectories.py          # ParticleTrajectorySimulator
│   │   ├── evaluation.py                     # Metrics (path length, MSD, etc.)
│   │   ├── modeling.py                       # CFD preprocessing
│   │   ├── pipeline.py                       # Orchestration utilities
│   │   ├── pyfr_simulation.py                # PyFR integration
│   │   └── visualization.py                  # Velocity field helpers
│   │
│   ├── cli/                                  # Conversational CLI
│   │   ├── __init__.py
│   │   ├── main.py                           # Entry point & dispatcher
│   │   ├── commands.py                       # Workflow handlers
│   │   └── chat.py                           # FlowChat (Rich + Questionary)
│   │
│   ├── interfaces/                           # Protocol definitions
│   │   ├── __init__.py
│   │   ├── config.py                         # SimulationConfig
│   │   ├── data_sources.py                   # VelocityFieldSource
│   │   ├── trajectories.py                   # TrajectoryResult
│   │   ├── launch_options.py                 # LaunchOptions
│   │   ├── modeling.py                       # ModelingConfig
│   │   ├── visualization.py                  # TrajectoryVisualizer
│   │   ├── cfd.py                            # CFDSimulator
│   │   └── pyfr.py                           # PyFR-specific config
│   │
│   ├── modeling/                             # Training infrastructure
│   │   ├── __init__.py
│   │   ├── datasets.py                       # CFDTrajectorySequenceDataset
│   │   └── trainers.py                       # Trainer (orchestration)
│   │
│   ├── networks/                             # Neural ODE models
│   │   ├── __init__.py
│   │   ├── cnf.py                            # CNFModel + ODEFunc
│   │   ├── variational_sde.py                # VariationalSDEModel
│   │   ├── mlp.py                            # MLP backbone
│   │   └── encoder.py                        # Trajectory encoder
│   │
│   ├── utils/                                # Shared utilities
│   │   ├── __init__.py
│   │   ├── paths.py                          # project_root(), resolve_data_path()
│   │   ├── config.py                         # load_config()
│   │   ├── load_config.py                    # Experiment-specific config loaders
│   │   ├── trajectory_io.py                  # load/save_trajectory_bundle()
│   │   ├── cache.py                          # Cache management
│   │   ├── orchestrator.py                   # ExperimentOrchestrator (YAML runner)
│   │   ├── console_gate.py                   # Interactive prompt detection
│   │   ├── prompt_sync.py                    # Session state persistence
│   │   └── modeling.py                       # Training utilities
│   │
│   ├── visualization/                        # Plotting & animation
│   │   ├── __init__.py
│   │   ├── trajectories.py                   # TrajectoryPlotter
│   │   ├── velocity_field.py                 # VelocityFieldPlotter
│   │   ├── training.py                       # plot_loss_curve(), etc.
│   │   └── viewer_taichi.py                  # 3D GPU-accelerated viewer
│   │
│   ├── workflows/                            # Standalone pipeline scripts
│   │   ├── __init__.py
│   │   ├── import_trajectories.py            # Particle simulation script
│   │   ├── train_cnf.py                      # CNF training script
│   │   ├── train_variational_sde.py          # SDE training script
│   │   ├── render_velocity_animations.py     # Animation generation
│   │   ├── render_velocity_field.py          # Static quiver plots
│   │   ├── prepare_demo_artifacts.py         # Synthetic data generator
│   │   ├── summarize_demo_artifacts.py       # Print bundle statistics
│   │   └── run_vsde_inference.py             # SDE sampling script
│   │
│   └── experiments/                          # Experiment definitions
│       ├── __init__.py
│       ├── demo-baseline.yaml                # Minimal example
│       ├── euler-vortex.yaml                 # 2D Euler vortex
│       ├── viscous-shock-tube.yaml           # Diffusion-dominated
│       ├── inc-cylinder.yaml                 # Cylinder flow
│       ├── velocity-animations.yaml          # Batch animation job
│       ├── docs/                             # Experiment documentation
│       └── scripts/                          # Experiment-specific helpers
│           ├── __init__.py
│           ├── train_cnf.py
│           └── train_variational_sde.py
│
├── scripts/                                  # Standalone analysis scripts
│   ├── analyze_bundle.py
│   ├── analyze_drift_components.py
│   ├── check_ckpt.py
│   ├── check_trajectories.py
│   ├── check_velocity_snapshots.py
│   ├── length_sweep.py
│   └── trajectory_diagnostics.py
│
├── data/                                     # Data directory
│   ├── 2d-euler-vortex/
│   │   ├── velocity/                         # Velocity snapshots (.npy)
│   │   ├── trajectories/                     # Simulation outputs
│   │   └── density/                          # Density fields (optional)
│   ├── 2d-viscous-shock-tube/
│   ├── demo/                                 # Demo synthetic data
│   └── empty.txt                             # Placeholder
│
├── cache/                                    # Transient outputs (git-ignored)
│   ├── checkpoints/                          # Model checkpoints
│   │   ├── 2d-euler-vortex_cnf/
│   │   ├── 2d-euler-vortex_vsde/
│   │   ├── 2d-viscous-shock-tube_cnf/
│   │   └── 2d-viscous-shock-tube_vsde/
│   ├── artifacts/                            # Plots, animations, reports
│   └── runtime/                              # Prompt states, temp files
│
├── docs/                                     # Documentation
│   └── flow_usage.md                         # Detailed walkthrough
│
├── tests/                                    # Unit tests (minimal)
│   └── cfd/                                  # CFD tests
│
└── old_code/                                 # Legacy code (reference only)
    ├── README.md
    ├── proposal.md
    └── ...
```

**Leaf Packages** (with comprehensive `__init__.py` mini-READMEs):

- `src/`
- `src/app/`
- `src/cfd/`
- `src/cli/`
- `src/interfaces/`
- `src/modeling/`
- `src/networks/`
- `src/utils/`
- `src/visualization/`
- `src/workflows/`
- `src/experiments/`
- `src/experiments/scripts/`

---

## Advanced Examples

### Example 1: End-to-End Workflow from Python

```python
import torch
from pathlib import Path
from src.utils.config import load_config
from src.utils.trajectory_io import load_trajectory_bundle, save_trajectory_bundle
from src.interfaces.data_sources import NpyVelocityFieldSource
from src.cfd.particle_trajectories import ParticleTrajectorySimulator
from src.modeling.datasets import CFDTrajectorySequenceDataset
from src.networks.cnf import CNFModel
from src.modeling.trainers import Trainer

# 1. Load configuration
config = load_config()
print(f"Simulation config: particles={config.trajectory_particles}, dt={config.trajectory_dt}")

# 2. Simulate trajectories
velocity_source = NpyVelocityFieldSource(config.velocity_dir)
simulator = ParticleTrajectorySimulator(velocity_source=velocity_source, solver="rk4")
result = simulator.simulate(
    n_particles=200,
    max_steps=100,
    dt=0.01,
)
save_trajectory_bundle(result, "output/simulated_trajectories.npz")
print(f"Simulated {result.positions.shape[0]} particles over {result.positions.shape[1]} steps")

# 3. Prepare dataset
dataset = CFDTrajectorySequenceDataset(
    trajectory_bundles=[result],
    sequence_length=20,
    normalize=True
)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
)

# 4. Train CNF model
model = CNFModel(dim=2, cond_dim=64, hidden_dim=128, depth=3)
trainer = Trainer(
    model=model,
    device="mps",  # "cuda" on GPU, "cpu" for CPU
    checkpoint_dir=Path("cache/checkpoints/my_cnf"),
)
trainer.train(
    dataloader,
    num_epochs=100,
    learning_rate=1e-3,
)
print(f"Best loss: {trainer.best_loss:.4f}")

# 5. Load best model
trainer.load_checkpoint()
model.eval()

# 6. Visualize
from src.visualization.trajectories import TrajectoryPlotter
plotter = TrajectoryPlotter(figsize=(12, 8), cmap="viridis")
fig = plotter.plot_trajectories(result, save_path="output/trajectories.png")
print("Saved plot to output/trajectories.png")
```

### Example 2: Custom Velocity Source

```python
import numpy as np
from src.interfaces.data_sources import VelocityFieldSource
from src.cfd.particle_trajectories import ParticleTrajectorySimulator

class AnalyticalVortexSource(VelocityFieldSource):
    """Analytical 2D vortex velocity field."""
  
    def query(self, x, y, z=None, t=0.0):
        """Return velocity at (x, y) for 2D vortex."""
        r_sq = x**2 + y**2
        # Avoid singularity at origin
        r_sq = np.maximum(r_sq, 1e-6)
  
        # Circulation velocity: v_theta = Gamma / (2*pi*r)
        v_theta = 1.0 / (2 * np.pi * np.sqrt(r_sq))
  
        # Convert to Cartesian
        vx = -v_theta * y / np.sqrt(r_sq)
        vy = v_theta * x / np.sqrt(r_sq)
  
        return np.stack([vx, vy], axis=-1)

# Use custom source
velocity_source = AnalyticalVortexSource()
simulator = ParticleTrajectorySimulator(velocity_source=velocity_source)
result = simulator.simulate(n_particles=100, max_steps=50, dt=0.01)
```

---

## Troubleshooting

### 1. "Config file not found at {path}"

**Cause**: `config.yml` missing or incorrect project structure.

**Solution**:

```bash
# Verify you're in the correct directory
ls config.yml  # Should exist in project root

# Check project root detection
python -c "from src.utils.paths import project_root; print(project_root())"
```

### 2. "RuntimeError: Context must be set before integrating the CNF"

**Cause**: Calling `model._integrate()` without setting context.

**Solution**:

```python
model.func.set_context(context_vector)  # Set first
z_t, log_prob = model._integrate(z0, context, tspan)
```

### 3. "ValueError: tspan must contain at least two time points"

**Cause**: Time span too short.

**Solution**:

```python
# Use at least 2 time points
tspan = torch.linspace(0, 1, 10)  # ✓ Good
# tspan = torch.linspace(0, 1, 1)  # ✗ Bad
```

### 4. GPU Out of Memory

**Cause**: Batch size too large or model too wide.

**Solutions**:

Reduce batch size, reduce model width/depth, or switch to CPU/MPS if CUDA memory is limited.

Prefer YAML pipelines? Run `pivo`, then choose `experiment` → `velocity-animations`.

### VSDE Inference Integrator Comparison

`src/workflows/run_vsde_inference.py` now exposes a `--integrator` flag (choices: `euler`, `improved_euler`, `rk4`, `dopri5`) so you can target different drift solvers while diffusion always uses Euler–Maruyama. Use `pivo`, then choose `experiment` → `integrator-comparison` to execute inference with each integrator back-to-back and inspect the overlays living under `cache/artifacts/integrator-comparison/<method>`.

The final step of the experiment runs `src/experiments/scripts/compare_integrators.py`, which produces `cache/artifacts/integrator-comparison/charts/comparison_mae.png` plus a summary JSON detailing the VSDE/CNF MAE per integrator.

## Project Layout

- `src/` – Core Python package (CLI, CFD utilities, hybrid models, visualization helpers).
- `docs/flow_usage.md` – Detailed walkthrough of both interfaces and automation tips.
- `src/experiments/` – YAML experiment definitions and helper scripts.
- `config.yml` – Simulation defaults shared by CLI/GUI.
- `requirements.txt` – Exact dependency pins reused by the packaging metadata.

## License

Released under the **MIT License**. See `LICENSE` for details.

---

## Citation

If you use PIVONet in your research, please cite:

```bibtex
@software{pivonet2025,
  title={PIVONet: Physics-Informed Variational ODE Networks},
  author={Cheung, Hei Shing and Lin, David and Long, Ethan},
  year={2025},
  url={https://github.com/HaysonC/ODE_CNF_thermal_motion},
}
```

## Questions?

- **Documentation**: See `docs/flow_usage.md` for detailed walkthrough
- **Issues**: https://github.com/HaysonC/ODE_CNF_thermal_motion/issues
- **Code**: Browse `src/` for well-documented source
