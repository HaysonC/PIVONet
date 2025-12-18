"""Streamlit front end for the Flow toolkit."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, TypeVar

import streamlit as st

from src.cli.commands import (
    InteractionChannel,
    run_import,
    run_velocity,
    run_visualize,
    run_modeling,
)
from src.interfaces.launch_options import LaunchOptions
from src.utils.load_config import load_config

T = TypeVar("T")


def main() -> None:
    st.set_page_config(page_title="PIVONet", layout="wide")
    config = load_config()

    channel = StreamlitChannel()

    st.title("PIVONet Control Center")
    st.write(
        "Use these forms to invoke trajectory simulation, visualization, or velocity field sketches in one place."
    )

    st.sidebar.header("Quick actions")
    st.sidebar.caption(f"Simulation version: {config.version}")
    command = st.sidebar.radio("Command", LaunchOptions.conversational_commands())

    if command == "import":
        _render_import_panel(channel, config)
    elif command == "visualize":
        _render_visualization_panel(channel)
    elif command == "velocity":
        _render_velocity_panel(channel)
    else:
        _render_modeling_panel(channel)


class StreamlitChannel(InteractionChannel):
    def __init__(self) -> None:
        self._last_paths: dict[str, Path] = {}

    def say(self, message: str) -> None:
        st.info(message)

    def success(self, message: str) -> None:
        st.success(message)

    def hint(self, option: LaunchOptions) -> None:
        st.caption(f"Try the CLI: `{option.command_hint()}`")

    def default_path(self, key: str) -> Path | None:
        return self._last_paths.get(key)

    def remember_path(self, key: str, path: Path | None) -> None:
        if path is not None:
            self._last_paths[key] = path


def _render_import_panel(channel: InteractionChannel, config) -> None:
    st.subheader("Import velocity snapshots")
    st.caption(
        "Simulate particle trajectories using the PyFR velocity snapshots stored in data/."
    )
    with st.form("import-form"):
        particles = st.number_input(
            "Particles", min_value=1, step=32, value=config.trajectory_particles
        )
        max_steps = st.number_input(
            "Max snapshots (0 = unlimited)", min_value=0, value=0, step=50
        )
        dt = st.number_input(
            "Integration timestep (dt)",
            min_value=0.001,
            value=config.trajectory_dt,
            format="%.4f",
        )
        output_path = st.text_input("Trajectory bundle output path (optional)")
        run_name = st.text_input("Run name (optional)")
        submitted = st.form_submit_button("Simulate import")
    if submitted:
        options = LaunchOptions(
            command="import",
            particles=int(particles),
            max_steps=None if int(max_steps) <= 0 else int(max_steps),
            dt=float(dt),
            diffusion_constant=config.diffusion_constant,
            output_path=_to_path(output_path),
            run_name=run_name.strip() or None,
        )
        result = _execute_command(channel, options, run_import)
        if result is not None:
            st.metric("Particles simulated", result.num_particles)
            st.metric("Steps", max(0, result.history.shape[0] - 1))


def _render_visualization_panel(channel: InteractionChannel) -> None:
    st.subheader("Trajectory visualization")
    st.caption(
        "Render one of your saved trajectory bundles into a publication-quality plot."
    )
    with st.form("visualize-form"):
        input_path = st.text_input("Trajectory bundle path (.npz/.npy)")
        max_particles = st.number_input(
            "Max particles to draw", min_value=10, value=200, step=10
        )
        output_path = st.text_input("Output image path (optional)")
        flow_overlay = st.checkbox("Overlay velocity field", value=True)
        submitted = st.form_submit_button("Render visualization")
    if submitted:
        if not input_path.strip():
            st.error("Please supply a trajectory bundle path.")
            return
        options = LaunchOptions(
            command="visualize",
            input_path=Path(input_path).expanduser(),
            max_particles=int(max_particles),
            output_path=_to_path(output_path),
            flow_overlay=flow_overlay,
        )
        _execute_command(channel, options, run_visualize)


def _render_velocity_panel(channel: InteractionChannel) -> None:
    st.subheader("Velocity field visualization")
    st.caption(
        "Sketch velocity snapshots saved as .npy files with an intuitive arrow plot."
    )
    with st.form("velocity-form"):
        input_path = st.text_input("Velocity field path (.npy)")
        sample_amount = st.number_input(
            "Number of velocity samples", min_value=1_000, value=50_000, step=5_000
        )
        output_path = st.text_input("Output image path (optional)")
        submitted = st.form_submit_button("Render velocity field")
    if submitted:
        if not input_path.strip():
            st.error("Please supply a velocity field path.")
            return
        options = LaunchOptions(
            command="velocity",
            input_path=Path(input_path).expanduser(),
            velocity_samples=int(sample_amount),
            output_path=_to_path(output_path),
        )
        _execute_command(channel, options, run_velocity)


def _render_modeling_panel(channel: InteractionChannel) -> None:
    st.subheader("Train hybrid encoder + CNF")
    st.caption(
        "Fit the variational encoder and CNF on a saved trajectory bundle with custom hyperparameters."
    )
    with st.form("modeling-form"):
        bundle_path = st.text_input("Trajectory bundle path (.npz/.npy)")
        latent_dim = st.number_input("Latent dimension", min_value=2, value=16, step=1)
        context_dim = st.number_input(
            "Context dimension", min_value=8, value=64, step=4
        )
        encoder_steps = st.number_input(
            "Encoder iterations", min_value=1, value=8, step=1
        )
        encoder_lr = st.number_input(
            "Encoder learning rate", min_value=1e-6, value=0.001, format="%.6f"
        )
        cnf_steps = st.number_input("CNF iterations", min_value=1, value=6, step=1)
        cnf_lr = st.number_input(
            "CNF learning rate", min_value=1e-6, value=0.0002, format="%.6f"
        )
        cnf_hidden = st.number_input(
            "CNF hidden units", min_value=32, value=128, step=8
        )
        run_tag = st.text_input("Run tag (optional)")
        submitted = st.form_submit_button("Train models")
    if submitted:
        if not bundle_path.strip():
            st.error("Please provide the trajectory bundle path to train on.")
            return
        options = LaunchOptions(
            command="model",
            model_input_path=Path(bundle_path).expanduser(),
            model_run_name=run_tag.strip() or None,
            encoder_latent_dim=int(latent_dim),
            encoder_context_dim=int(context_dim),
            encoder_steps=int(encoder_steps),
            encoder_lr=float(encoder_lr),
            cnf_steps=int(cnf_steps),
            cnf_lr=float(cnf_lr),
            cnf_hidden_dim=int(cnf_hidden),
        )
        outcome = _execute_command(channel, options, run_modeling)
        if outcome:
            metrics = outcome.result.metrics
            st.metric("Encoder loss", f"{metrics.get('encoder_loss', 0.0):.6f}")
            st.metric("CNF loss", f"{metrics.get('cnf_loss', 0.0):.6f}")
            st.caption(f"Checkpoints saved to: {outcome.cache_dir}")


def _execute_command(
    channel: InteractionChannel,
    options: LaunchOptions,
    handler: Callable[[LaunchOptions, InteractionChannel], T],
) -> T | None:
    try:
        return handler(options, channel)
    except Exception as error:  # pragma: no cover - user triggered fault
        st.error(f"{error.__class__.__name__}: {error}")
        channel.hint(options)
        return None


def _to_path(value: str | None) -> Path | None:
    candidate = (value or "").strip()
    if not candidate:
        return None
    return Path(candidate).expanduser()


if __name__ == "__main__":
    main()
