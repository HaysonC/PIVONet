"""Shared utilities for paths, configuration, I/O, and orchestration.

===================================================================================
OVERVIEW
===================================================================================
Provides common helper functions and utilities used across PIVONet:
  - Path resolution (project root, data directories)
  - Configuration loading (YAML parsing)
  - Trajectory I/O (serialization, bundle management)
  - Experiment orchestration (YAML-driven pipeline execution)
  - Console interaction (interactive prompts)
  - Caching (run-specific data management)

===================================================================================
SUBMODULE STRUCTURE
===================================================================================

paths.py:
    project_root() → project absolute path
    data_root() → data/ directory
    resolve_data_path(*segs) → normalized path under data/
    All paths cached as module-level variables for performance

config.py:
    load_config(path=None) → SimulationConfig from YAML
    Parses config.yml for simulation parameters
    Handles defaults and type coercion

load_config.py:
    Variant load functions for specific experiment configs
    load_experiment_config() → ExperimentConfig

trajectory_io.py:
    trajectories_dir() → directory where .npz bundles stored
    load_trajectory_bundle(path) → TrajectoryResult
    save_trajectory_bundle(result, path) → Path
    list_trajectory_files(pattern) → list of Path objects
    Handles .npz serialization format

cache.py:
    cache_root() → cache/ directory
    resolve_cache_path(*segs) → normalized path under cache/
    create_run_cache(run_id) → timestamped run directory
    Manages checkpoint and artifact storage

orchestrator.py:
    ExperimentStep, ExperimentSpec (dataclasses)
    ExperimentOrchestrator - Main YAML pipeline executor
    Loads experiments from YAML, executes steps sequentially
    Progress tracking, timing, error handling

console_gate.py:
    is_prompt_active() → bool (check if interactive)
    Utilities for conditional prompt behavior in CI environments

prompt_sync.py:
    Utilities for state persistence across interactive sessions
    Remembers user choices and paths

modeling.py:
    TrainingOutcome (dataclass) - encapsulates train results
    modeling_config_from_options() → ModelingConfig
    train_from_bundle() → orchestrate training workflow

===================================================================================
KEY TYPES
===================================================================================

SimulationConfig:
    version: str
    diffusion_constant: float
    data_root: Path
    velocity_subdir: str
    trajectory_particles: int
    trajectory_steps: int | None
    trajectory_dt: float

TrajectoryResult:
    positions: ndarray(n_particles, n_steps, dim)
    velocities: ndarray(n_particles, n_steps, dim)
    times: ndarray(n_steps,)
    metadata: dict

ExperimentStep:
    name: str
    description: str
    script: str (path to .py script)
    params: dict[str, Any]

ExperimentOrchestrator:
    Loads YAML specs, executes steps, tracks progress
    Main entry point: run_experiment(slug, overrides={})

===================================================================================
DATA FLOW DIAGRAM
===================================================================================

    Application Startup
            ↓
    load_config() → SimulationConfig
            ↓
    project_root() → project path cache
    data_root() → data path cache
            ↓
    User Interaction (CLI/GUI)
            ↓
    resolve_data_path() → normalized paths
    load_trajectory_bundle() ← I/O operations
    save_trajectory_bundle() ↓
            ↓
    ExperimentOrchestrator.run()
            ↓
    YAML parsing → ExperimentSpec
            ↓
    For each step:
      - create_run_cache() → run directory
      - subprocess.run(step.script, cwd=project_root())
      - Track timing and result
            ↓
    Summary report, save artifacts

===================================================================================
USAGE PATTERNS
===================================================================================

# Load configuration
from src.utils.config import load_config
config = load_config()
print(f"Particles: {config.trajectory_particles}")

# Resolve paths
from src.utils.paths import project_root, resolve_data_path
proj = project_root()
data_path = resolve_data_path("2d-euler-vortex", "velocity", create=True)

# I/O trajectories
from src.utils.trajectory_io import load_trajectory_bundle, save_trajectory_bundle
result = load_trajectory_bundle("data/traj.npz")
save_trajectory_bundle(result, "output/modified.npz")

# Run experiment
from src.utils.orchestrator import ExperimentOrchestrator
orchestrator = ExperimentOrchestrator()
orchestrator.run_experiment("demo-baseline", overrides={"particles": 200})

===================================================================================
MEMORY MANAGEMENT
===================================================================================

Paths: Computed once at module load, cached as module-level variables.
No performance penalty for repeated project_root() calls.

Configs: load_config() creates new instance each time (allows hot-reloading).
Caller should cache if needed for performance.

Trajectory bundles: Loaded into memory via numpy (no lazy evaluation).
For large datasets (>1GB), consider streaming or batch processing.

Cache directories: Automatically cleaned up if prune_history() called.
Default retention: keep 5 most recent runs per experiment.

===================================================================================
ERROR HANDLING
===================================================================================

FileNotFoundError:
    - config.yml missing → check project_root() is correct
    - trajectory .npz missing → verify path via list_trajectory_files()

ValueError:
    - Invalid YAML syntax → validate experiment .yaml against schema
    - resolve_data_path with invalid segments → check path construction

RuntimeError:
    - Step subprocess fails → check step.script exists and is executable
    - YAML load fails → ensure valid YAML indentation and syntax

===================================================================================
"""

from .cache import cache_root, create_run_cache, resolve_cache_path
from .paths import data_root, project_root, resolve_data_path
from .trajectory_io import (
    list_trajectory_files,
    load_trajectory_bundle,
    trajectories_dir,
)

__all__ = [
    "project_root",
    "data_root",
    "resolve_data_path",
    "trajectories_dir",
    "list_trajectory_files",
    "load_trajectory_bundle",
    "cache_root",
    "resolve_cache_path",
    "create_run_cache",
]
