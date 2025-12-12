"""Visualization helpers for trajectories, velocity fields, and training metrics.

===================================================================================
OVERVIEW
===================================================================================
Provides plotting utilities for analyzing and presenting simulation results:
  - Trajectory path visualization (2D/3D)
  - Velocity field quiver plots and animations
  - Training loss curves and metric monitoring
  - Interactive Taichi-based 3D viewer for real-time exploration

Main use: Post-processing and analysis of CFD+ML pipeline outputs.

===================================================================================
SUBMODULE STRUCTURE
===================================================================================

trajectories.py:
    TrajectoryPlotter - Main class for trajectory visualization
    Methods:
      - plot_trajectories(result, save_path) → matplotlib figure
      - animate_trajectories(result, output_path) → GIF/MP4
      - plot_phase_portrait(positions, velocities) → scatter plot
      - compute_trajectory_statistics() → dict of metrics

velocity_field.py:
    VelocityFieldPlotter - Velocity field rendering
    Methods:
      - plot_quiver(velocity, positions) → matplotlib figure
      - plot_streamlines(velocity) → streamline plot
      - plot_magnitude(velocity) → heatmap
      - animate_time_evolution() → GIF sequence

training.py:
    plot_loss_curve(history) → loss trajectory plot
    plot_metric_grid(metrics_dict) → multi-subplot grid
    plot_convergence(losses, moving_avg_window) → smoothed curve
    plot_validation_split() → train/val comparison

viewer_taichi.py:
    GPU-accelerated 3D interactive viewer
    Features:
      - Play/pause particle animation
      - Camera orbiting and zoom
      - Per-particle coloring
      - Real-time CNF model sampling (if checkpoint available)
      - Performance: ~60 FPS for 10k particles

===================================================================================
KEY TYPES
===================================================================================

TrajectoryPlotter:
    Attributes:
      - figsize: (width, height) in inches
      - dpi: resolution
      - cmap: matplotlib colormap for particle colors
      - alpha: transparency
    Methods:
      - plot_trajectories() → Figure
      - animate_trajectories() → str (path)

VelocityFieldPlotter:
    Attributes:
      - vector_stride: decimation for quiver (reduce clutter)
      - scale: arrow scaling factor
      - cmap: colormap for magnitude
    Methods:
      - plot_quiver() → Figure
      - animate_time_evolution() → str (path)

===================================================================================
DATA FLOW DIAGRAM
===================================================================================

    TrajectoryResult
            │
    ┌───────┴──────────┐
    │                  │
    ▼                  ▼
TrajectoryPlotter  plot_loss_curve
    │                  │
    ├─ positions   ├─ training metrics
    ├─ velocities  ├─ loss history
    ├─ times       └─ convergence check
    └─ metadata
            │                  │
    ┌───────▼──────────┬───────▼────────┐
    │                  │                │
    ▼                  ▼                ▼
  matplotlib      matplotlib         Animation/GIF
  static figures  training plots      files
            │                  │                │
            └──────────┬───────┴────────┬───────┘
                       ▼
                 User view/export

===================================================================================
USAGE PATTERNS
===================================================================================

# Plot trajectory paths
from src.visualization.trajectories import TrajectoryPlotter
plotter = TrajectoryPlotter(figsize=(12, 8))
fig = plotter.plot_trajectories(result, save_path="output.png")

# Animate trajectories
gif_path = plotter.animate_trajectories(result, output_path="anim.gif", fps=24)

# Plot training curves
from src.visualization.training import plot_loss_curve
fig = plot_loss_curve(trainer.loss_history)
fig.savefig("training_loss.png")

# Interactive 3D viewer
from src.visualization.viewer_taichi import main as launch_viewer
launch_viewer(dataset="2d-euler-vortex", checkpoint_path="model.pt")

# Velocity field visualization
from src.visualization.velocity_field import VelocityFieldPlotter
velocity_plotter = VelocityFieldPlotter(vector_stride=3)
fig = velocity_plotter.plot_quiver(velocity_data, mesh_points)

===================================================================================
MEMORY MANAGEMENT
===================================================================================

Matplotlib figures: Held in memory until .savefig() or .show() called.
For batch visualization, call plt.close() after saving to free memory.

Animations: Rendered to disk as GIF/MP4. Consider frame rate and duration
to avoid massive file sizes. Typical sizes:
  - 100 particles, 100 steps, 24 FPS → 500KB GIF
  - 1000 particles, 1000 steps → 50MB GIF

Taichi viewer: GPU memory proportional to particle count. Recommended limits:
  - <10k particles: smooth 60 FPS on modest GPU
  - 10k-100k particles: 30-60 FPS with stride decimation
  - >100k particles: consider sub-sampling

===================================================================================
CONSTRAINTS & ASSUMPTIONS
===================================================================================

1. Trajectories assumed bounded (no extreme coordinates)
2. Time arrays assumed strictly increasing
3. Velocity fields expected on regular or mesh-specified grids
4. Animations expect consistent frame dimensions
5. For Taichi viewer: NVIDIA CUDA or Apple MPS required for GPU acceleration

===================================================================================
ERROR HANDLING
===================================================================================

FileNotFoundError:
    - Output path not writable → check parent directory permissions

ValueError:
    - Empty trajectory array → verify result.positions is non-empty
    - Incompatible dimensions → check position/velocity shape consistency

RuntimeError:
    - GPU out of memory (Taichi) → reduce particle count or stride
    - Animation encoder missing → install ffmpeg for MP4 support

===================================================================================
"""

from .trajectories import TrajectoryPlotter
from .training import plot_loss_curve, plot_metric_grid
from .velocity_field import VelocityFieldPlotter

__all__ = [
	"TrajectoryPlotter",
	"VelocityFieldPlotter",
	"plot_loss_curve",
	"plot_metric_grid",
]
