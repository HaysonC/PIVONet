"""Modular command handlers for the Flow CLI."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Protocol

from ..interfaces.launch_options import LaunchOptions
from ..utils.paths import resolve_data_path
from ..utils.trajectory_io import load_trajectory_bundle
from ..utils.modeling import (
    TrainingOutcome,
    modeling_config_from_options,
    train_from_bundle,
)
from ..visualization import TrajectoryPlotter, VelocityFieldPlotter
from ..utils.paths import project_root


class InteractionChannel(Protocol):
    def say(self, message: str) -> None:  # pragma: no cover - simple logging interface
        ...

    def success(
        self, message: str
    ) -> None:  # pragma: no cover - simple logging interface
        ...

    def hint(
        self, option: LaunchOptions
    ) -> None:  # pragma: no cover - simple logging interface
        ...

    def default_path(
        self, key: str
    ) -> Path | None:  # pragma: no cover - persistence helper
        ...

    def remember_path(
        self, key: str, path: Path | None
    ) -> None:  # pragma: no cover - persistence helper
        ...


def _default_plot_path(input_path: str, subdir: str) -> Path:
    base = Path(input_path).stem or "output"
    filename = f"{base}.png"
    return resolve_data_path("cfd", subdir, filename, create=True)


def run_import(options: LaunchOptions, channel: InteractionChannel) -> Path:
    assert options.import_data_source, "Import requires a dataset source folder."
    assert options.import_data_flow, "Import requires a flow name."

    source = Path(options.import_data_source).expanduser().resolve()
    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"Dataset folder does not exist: {source}")

    velocity_dir = source / "velocity"
    if not velocity_dir.exists() or not velocity_dir.is_dir():
        raise ValueError(
            f"Dataset folder must contain a velocity/ subfolder: {source}"
        )

    dest_flow_dir = (project_root() / "data" / options.import_data_flow).resolve()
    channel.say(
        f"Copying dataset into data/{options.import_data_flow}/ ..."
    )

    dest_flow_dir.parent.mkdir(parents=True, exist_ok=True)
    if dest_flow_dir.exists():
        if options.import_data_overwrite:
            shutil.rmtree(dest_flow_dir)
        else:
            raise FileExistsError(
                f"Target already exists: {dest_flow_dir} (re-run with overwrite enabled)"
            )

    shutil.copytree(source, dest_flow_dir)
    channel.success(
        f"Imported dataset to [bold green]{dest_flow_dir}[/] (expects velocity/ inside)."
    )
    channel.hint(options)
    channel.remember_path("import_data_source", source)
    return dest_flow_dir


def run_visualize(options: LaunchOptions, channel: InteractionChannel) -> None:
    assert options.input_path, "Visualize command requires an input trajectory bundle."
    channel.say("Rendering your saved trajectory bundle now.")
    bundle = load_trajectory_bundle(str(options.input_path))
    plotter = TrajectoryPlotter(max_particles=options.max_particles)
    output_path = options.output_path or _default_plot_path(
        str(options.input_path), "plots"
    )
    artifact = plotter.plot(bundle, output_path=output_path, show=False, title=None)
    channel.success(f"Generated comparison plot at [bold green]{artifact.path}[/].")
    channel.hint(options)
    channel.remember_path("trajectory_bundle", options.input_path)


def run_velocity(options: LaunchOptions, channel: InteractionChannel) -> None:
    """Deprecated: retained for Streamlit UI compatibility."""
    assert options.input_path, "Velocity plot command needs an input .npy file."
    channel.say("Sampling the velocity field and sketching the flow.")
    plotter = VelocityFieldPlotter(sample_points=options.velocity_samples)
    output_path = options.output_path or _default_plot_path(
        str(options.input_path), "velocity_plots"
    )
    artifact = plotter.plot_from_file(
        str(options.input_path), output_path=output_path, show=False, title=None
    )
    channel.success(f"Velocity visualization saved to [bold green]{artifact.path}[/].")
    channel.hint(options)
    channel.remember_path("velocity_field", options.input_path)


def run_modeling(
    options: LaunchOptions, channel: InteractionChannel
) -> TrainingOutcome:
    assert options.model_input_path or options.input_path, (
        "Training requires a saved trajectory bundle path."
    )
    bundle = options.model_input_path or options.input_path
    assert bundle is not None
    channel.say(
        "Training the diffusion encoder + CNF models with your chosen hyperparameters."
    )
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


def run_import_model(options: LaunchOptions, channel: InteractionChannel) -> Path:
    assert options.import_model_source, "Import model requires a source path."

    source = Path(options.import_model_source).expanduser().resolve()
    checkpoints_root = (project_root() / "cache" / "checkpoints").resolve()

    # If target is omitted, treat source as a bundle containing multiple checkpoint folders.
    target_dir = (
        (checkpoints_root / options.import_model_target).resolve()
        if options.import_model_target
        else checkpoints_root
    )

    channel.say("Importing pretrained checkpoints into cache/checkpoints...")

    if not source.exists():
        raise FileNotFoundError(f"Source path does not exist: {source}")

    checkpoints_root.mkdir(parents=True, exist_ok=True)

    if options.import_model_target is None:
        if not source.is_dir():
            raise ValueError(
                "Bundle import requires a directory source (containing one or more checkpoint folders)."
            )
        # Copy every immediate child directory into cache/checkpoints/<child>.
        copied: list[Path] = []
        for child in sorted(source.iterdir()):
            if not child.is_dir():
                continue
            dest = (checkpoints_root / child.name).resolve()
            if dest.exists():
                if options.import_model_overwrite:
                    shutil.rmtree(dest)
                else:
                    continue
            shutil.copytree(child, dest)
            copied.append(dest)

        if not copied:
            raise ValueError(
                f"No checkpoint folders found under {source}. Expected subfolders like '2d-euler-vortex_cnf'."
            )
        channel.success(
            "Imported checkpoint folders:\n" + "\n".join(f"- {p}" for p in copied)
        )
        channel.hint(options)
        return checkpoints_root

    # Single-target import
    assert options.import_model_target, "Import model requires a target name for single-folder imports."
    if target_dir.exists() and options.import_model_overwrite:
        shutil.rmtree(target_dir)

    if source.is_dir():
        if target_dir.exists() and not options.import_model_overwrite:
            raise FileExistsError(
                f"Target already exists: {target_dir} (re-run with overwrite enabled)"
            )
        shutil.copytree(source, target_dir)
    else:
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target_dir / source.name)

    channel.success(f"Imported model assets to [bold green]{target_dir}[/].")
    channel.hint(options)
    return target_dir
