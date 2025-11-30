"""Unified PIVO entry point that greets users then launches the CLI or GUI."""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Sequence

import questionary

from .utils.config import load_config
from .utils.paths import project_root
from .utils.console_gate import prompt_gate


def welcome_message(version: str) -> str:
    return f"""
===========================================================
PIVONet – Physics-Informed Variational ODE Networks
© 2025 Hayson Cheung, David Lin, Ethan Long

Version: {version}

PIVONet is an open-source tool for simulating and modeling physical systems using variational ordinary differential equations.

We invite you to explore its capabilities and contribute to its development.

PIVONet uses a dual approach for diffusion-advection simulations: 

  - A CNF model trained via neural ODE techniques for efficient trajectory prediction.
  - A varieational encoder to model diffusion processes.

We hope that this package empowers a end-to-end pipeline from data ingestion, simulation, evaluation, to visualization.
===========================================================
"""


def _launch_streamlit_app() -> None:
    script = project_root() / "src" / "app" / "ui.py"
    print("Starting the Streamlit UI...")
    try:
        subprocess.run(["streamlit", "run", str(script)], cwd=project_root(), check=False)
    except FileNotFoundError as error:  # pragma: no cover - streaming UI dependency
        raise RuntimeError("Streamlit is not installed in the current environment.") from error


def _parse_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gui", action="store_true", help="Launch the graphical Streamlit UI and exit.")
    parser.add_argument("--cli", action="store_true", help="Force the conversational CLI (default).")
    parser.add_argument("--no-banner", action="store_true", help="Suppress the ASCII welcome message.")
    return parser.parse_known_args(argv)


def _run_cli(extra_args: Sequence[str]) -> None:
    from .cli import main as cli_entry  # local import to avoid circular dependency

    cli_args = ["--skip-banner", *extra_args]
    cli_entry.main(cli_args)


def main(argv: Sequence[str] | None = None) -> None:
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    parsed, passthrough = _parse_args(raw_args)
    config = load_config()
    if not parsed.no_banner:
        print(welcome_message(config.version))
    if parsed.gui:
        _launch_streamlit_app()
        return

    if parsed.cli or passthrough:
        _run_cli(passthrough)
        return

    with prompt_gate():
        choice = questionary.select(
            "Select interface:",
            choices=[
                questionary.Choice(title="Command-Line (CLI)", value="cli"),
                questionary.Choice(title="Graphical (GUI)", value="gui"),
            ],
        ).ask()

    if choice == "gui":
        _launch_streamlit_app()
    else:
        _run_cli(passthrough)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
