"""Modular command handlers for the Flow CLI."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np

from ..cfd.particle_trajectories import ParticleTrajectorySimulator
from ..interfaces.data_sources import NpyVelocityFieldSource
from ..interfaces.launch_options import LaunchOptions
from ..interfaces.trajectories import TrajectoryResult
from ..utils.config import load_config
from ..utils.paths import resolve_data_path
from ..utils.trajectory_io import load_trajectory_bundle
from ..utils.modeling import TrainingOutcome, modeling_config_from_options, train_from_bundle
from ..visualization import TrajectoryPlotter, VelocityFieldPlotter


class InteractionChannel(Protocol):
    def say(self, message: str) -> None:  # pragma: no cover - simple logging interface
        ...

    def success(self, message: str) -> None:  # pragma: no cover - simple logging interface
        ...

    def hint(self, option: LaunchOptions) -> None:  # pragma: no cover - simple logging interface
        ...

    def default_path(self, key: str) -> Path | None:  # pragma: no cover - persistence helper
        ...

    def remember_path(self, key: str, path: Path | None) -> None:  # pragma: no cover - persistence helper
        ...


def _ensure_npz(path: Path) -> Path:
    return path if path.suffix == ".npz" else path.with_suffix(".npz")


def _default_trajectory_path(particles: int) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    filename = f"trajectories_n{particles}_{timestamp}.npz"
    return resolve_data_path("cfd", "trajectories", filename, create=True)


def _default_plot_path(input_path: str, subdir: str) -> Path:
    base = Path(input_path).stem or "output"
    filename = f"{base}.png"
    return resolve_data_path("cfd", subdir, filename, create=True)


def _monitor_trajectory(result: TrajectoryResult, channel: InteractionChannel) -> None:
    steps = result.history.shape[0] - 1
    channel.say(f"Simulated {result.num_particles} particles for {steps} timesteps.")


def run_import(options: LaunchOptions, channel: InteractionChannel) -> TrajectoryResult:
    channel.say("I will import velocity snapshots and start the particle simulation.")
    config = load_config()
    source = NpyVelocityFieldSource(config.velocity_dir)
    simulator = ParticleTrajectorySimulator(
        velocity_source=source,
        diffusion_coefficient=config.diffusion_constant,
        dt=options.dt,
        seed=None,
    )
    result = simulator.simulate(num_particles=options.particles, max_steps=options.max_steps)
    _monitor_trajectory(result, channel)

    if options.output_path:
        requested = Path(options.output_path).expanduser()
        output_path = requested if requested.is_absolute() else resolve_data_path(*requested.parts, create=True)
    else:
        output_path = _default_trajectory_path(options.particles)
    output_path = _ensure_npz(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, history=result.history, timesteps=np.asarray(result.timesteps))

    channel.success(f"History saved to [bold green]{output_path}[/].")
    channel.hint(options)
    channel.remember_path("trajectory_bundle", output_path)
    return result


def run_visualize(options: LaunchOptions, channel: InteractionChannel) -> None:
    assert options.input_path, "Visualize command requires an input trajectory bundle."
    channel.say("Rendering your saved trajectory bundle now.")
    bundle = load_trajectory_bundle(str(options.input_path))
    plotter = TrajectoryPlotter(max_particles=options.max_particles)
    output_path = options.output_path or _default_plot_path(str(options.input_path), "plots")
    artifact = plotter.plot(bundle, output_path=output_path, show=False, title=None)
    channel.success(f"Generated comparison plot at [bold green]{artifact.path}[/].")
    channel.hint(options)
    channel.remember_path("trajectory_bundle", options.input_path)


def run_velocity(options: LaunchOptions, channel: InteractionChannel) -> None:
    assert options.input_path, "Velocity plot command needs an input .npy file."
    channel.say("Sampling the velocity field and sketching the flow.")
    plotter = VelocityFieldPlotter(sample_points=options.velocity_samples)
    output_path = options.output_path or _default_plot_path(str(options.input_path), "velocity_plots")
    artifact = plotter.plot_from_file(str(options.input_path), output_path=output_path, show=False, title=None)
    channel.success(f"Velocity visualization saved to [bold green]{artifact.path}[/].")
    channel.hint(options)
    channel.remember_path("velocity_field", options.input_path)


def run_modeling(options: LaunchOptions, channel: InteractionChannel) -> TrainingOutcome:
    assert options.model_input_path or options.input_path, "Training requires a saved trajectory bundle path."
    bundle = options.model_input_path or options.input_path
    assert bundle is not None
    channel.say("Training the diffusion encoder + CNF models with your chosen hyperparameters.")
    config = modeling_config_from_options(options)
    outcome = train_from_bundle(bundle, config=config)
    metrics = outcome.result.metrics
    details = (
        f"encoder loss={metrics.get('encoder_loss', float('nan')):.4f}, "
        f"cnf loss={metrics.get('cnf_loss', float('nan')):.4f}"
    )
    channel.success(
        f"Training complete. {details} Checkpoints saved in [bold green]{outcome.cache_dir}[/]."
    )
    channel.hint(options)
    channel.remember_path("trajectory_bundle", Path(bundle))
    return outcome
