"""Unified PIVO entry point that greets users then launches the CLI or GUI."""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Sequence

from rich.console import Console
from rich.panel import Panel
from rich.style import Style
from rich.text import Text
import questionary

console = Console()

from .utils.config import load_config
from .utils.paths import project_root
from .utils.console_gate import prompt_gate


def welcome_message(version: str) -> Text:
    banner_lines = [
        "██████╗ ██╗██╗   ██╗ ██████╗ ",
        "██╔══██╗██║██║   ██║██╔═══██╗",
        "██████╔╝██║██║   ██║██║   ██║",
        "██╔═══╝ ██║╚██╗ ██╔╝██║   ██║",
        "██║     ██║ ╚████╔╝ ╚██████╔╝",
        "╚═╝     ╚═╝  ╚═══╝   ╚═════╝ ",
    ]
    gradient = [
        "#C8A2FF",  # lavender
        "#B7AFFF",
        "#A6BBFF",
        "#96C8FF",
        "#85D4FF",  # light blue peak
        "#96C8FF",
        "#A6BBFF",
        "#B7AFFF",
        "#C8A2FF",  # back to lavender
    ]

    text = Text()
    for line in banner_lines:
        for idx, char in enumerate(line):
            color = gradient[idx % len(gradient)]
            text.append(char, Style(color=color, bold=True))
        text.append("\n")
    text.append(
        "Workflow Orchestrator for  PIVO – Physics-Informed Variational ODE \n",
        Style(color="cyan", bold=True),
    )
    text.append(f"Version: {version}\n", Style(color="bright_cyan"))
    text.append(
        "© 2025 Hayson Cheung, David Lin, Ethan Long\n\n", Style(color="grey70")
    )
    text.append(
        "PIVO is an open-source tool for simulating and modeling physical systems using variational ordinary differential equations.\n",
        Style(color="white"),
    )
    text.append(
        "We invite you to explore its capabilities and contribute to its development.\n",
        Style(color="white"),
    )
    text.append(
        "PIVO pairs a CNF model for trajectory prediction with a variational encoder for diffusion processes.\n",
        Style(color="white"),
    )
    text.append(
        "Use it to orchestrate data ingestion, simulation, evaluation, and visualization workflows.\n",
        Style(color="white"),
    )
    return text


def _launch_streamlit_app() -> None:
    script = project_root() / "src" / "app" / "ui.py"
    print("Starting the Streamlit UI...")
    try:
        subprocess.run(
            ["streamlit", "run", str(script)], cwd=project_root(), check=False
        )
    except FileNotFoundError as error:  # pragma: no cover - streaming UI dependency
        raise RuntimeError(
            "Streamlit is not installed in the current environment."
        ) from error


def _parse_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--gui", action="store_true", help="Launch the graphical Streamlit UI and exit."
    )
    parser.add_argument(
        "--cli", action="store_true", help="Launch the conversational CLI (default)."
    )
    parser.add_argument(
        "--no-banner", action="store_true", help="Suppress the ASCII welcome message."
    )
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
        console.print(welcome_message(config.version))
    if parsed.gui:
        _launch_streamlit_app()
        return

    if parsed.cli or passthrough:
        _run_cli(passthrough)
        return

    try:
        with prompt_gate():
            console.print(
                Panel(
                    "[bold yellow]Warning:[/bold yellow]\n"
                    "The GUI mode is unstable and experimental. It may lead to crashes or unexpected behavior.\n"
                    "For the safest and most robust experience, choose **CLI**.",
                    border_style="yellow",
                )
            )

            choice = questionary.select(
                "Select interface:",
                choices=[
                    questionary.Choice(
                        title="[CLI] Command-Line (recommended)", value="cli"
                    ),
                    questionary.Choice(
                        title="[GUI] Graphical - not recommended", value="gui"
                    ),
                ],
            ).ask()
    except KeyboardInterrupt:
        console.print("\n[red]Aborted by user.[/red]")
        sys.exit(0)

    if choice == "gui":
        _launch_streamlit_app()
    else:
        _run_cli(passthrough)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
