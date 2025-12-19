"""Conversational entry point for Flow's CLI experience."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import questionary
import yaml

from ..interfaces.config import SimulationConfig
from ..interfaces.launch_options import LaunchOptions
from ..utils.config import load_config
from ..utils.console_gate import prompt_gate
from ..main import welcome_message
from .chat import FlowChat
from .commands import run_import, run_visualize, run_modeling, run_import_model
from ..utils.orchestrator import ExperimentOrchestrator

# The dev options command is hidden from the conversational CLI for now.
# CLI_COMMANDS = ("import", "visualize", "model", "experiment")

CLI_COMMANDS = ("import", "experiment")

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PIVO CLI entry point", add_help=True)
    parser.add_argument(
        "--run-experiment", metavar="SLUG", help="Run a predefined experiment and exit."
    )
    parser.add_argument(
        "--list-experiments",
        action="store_true",
        help="List available experiments and exit.",
    )
    parser.add_argument(
        "--experiments-dir", default=None, help="Override the experiments directory."
    )
    parser.add_argument(
        "--dev-mode",
        action="store_true",
        help="Enable development shortcuts in the CLI experience.",
    )
    parser.add_argument(
        "--skip-banner",
        action="store_true",
        help="Suppress the startup banner (useful for scripts).",
    )
    parser.add_argument(
        "--progress-mode",
        choices=("auto", "bars", "plain"),
        default="plain",
        help="Unified progress style for CLI + experiments (propagated via FLOW_PROGRESS_MODE).",
    )
    return parser.parse_args(argv)


def run_conversational_cli(
    config: SimulationConfig,
    *,
    dev_mode: bool = False,
    progress_mode: str | None = None,
) -> None:
    """Launch the Flow CLI in a question-and-answer style."""

    chat = FlowChat()
    chat.greet()

    while True:
        command = _choose_command()
        if not command or command == "exit":
            chat.say("All right—come back anytime for more Flow simulations!")
            break

        try:
            if command == "experiment":
                _run_experiment_menu(chat, progress_mode)
                continue
            options = _collect_options(command, chat, config)
        except KeyboardInterrupt:
            chat.say("Command canceled. Returning to the main menu.")
            continue

        try:
            if command == "import":
                if options.command == "import-model":
                    run_import_model(options, chat)
                else:
                    run_import(options, chat)
            elif command == "visualize":
                run_visualize(options, chat)
            elif command == "model":
                run_modeling(options, chat)
        except Exception as error:  # pragma: no cover - interactive shell
            chat.wrap_error(error, options)


def _choose_command() -> str | None:
    choices = [*CLI_COMMANDS, "exit"]
    with prompt_gate():
        return questionary.select("What would you like to do?", choices=choices).ask()


def _collect_options(
    command: str, chat: FlowChat, config: SimulationConfig
) -> LaunchOptions:
    if command == "import":
        return _import_menu_options(chat, config)
    if command == "visualize":
        return _visualize_options(chat)
    if command == "model":
        return _modeling_options(chat)
    raise ValueError(f"Unsupported command: {command}")


def _import_menu_options(chat: FlowChat, config: SimulationConfig) -> LaunchOptions:
    """Import menu entry.

    Provides two import paths under the single `import` command:
    - Flow data: copy a downloaded dataset folder (must include velocity/) into data/<flow>/.
    - Model checkpoints: copy downloaded pretrained checkpoints into cache/checkpoints.
    """

    with prompt_gate():
        choice = questionary.select(
            "What would you like to import?",
            choices=[
                questionary.Choice(title="Flow data (copy velocity snapshots)", value="data"),
                questionary.Choice(title="Model checkpoints (pretrained)", value="model"),
            ],
        ).ask()

    if choice == "model":
        return _import_model_options(chat)
    return _import_options(chat, config)


def _import_flow_targets() -> list[str]:
    """Suggest flow names based on existing data folders and experiment yamls."""

    targets: set[str] = set()

    data_root = Path("data")
    if data_root.exists():
        for child in data_root.iterdir():
            if child.is_dir() and not child.name.startswith("."):
                targets.add(child.name)

    experiments_dir = Path("src/experiments")
    if experiments_dir.exists():
        for yaml_path in sorted(experiments_dir.glob("*.yaml")):
            try:
                text = yaml_path.read_text(encoding="utf-8")
            except Exception:
                continue
            # Cheap parse: collect occurrences of `data/<flow>/velocity`.
            for fragment in text.split("data/")[1:]:
                flow = fragment.split("/", 1)[0].strip()
                if not flow:
                    continue
                if all(c.isalnum() or c in {"-", "_"} for c in flow):
                    targets.add(flow)

    return sorted(targets)


def _import_options(chat: FlowChat, config: SimulationConfig) -> LaunchOptions:
    chat.say(
        "Point me at a dataset folder (it must contain a velocity/ subfolder). "
        "I'll copy it into this repo under data/<flow-name>/."
    )

    while True:
        source_path = _ask_path(
            chat,
            "Path to dataset folder (contains velocity/)",
            required=True,
            default_key="import_data_source",
        )
        assert source_path is not None
        source = source_path.expanduser().resolve()
        velocity_dir = source / "velocity"
        if velocity_dir.exists() and velocity_dir.is_dir():
            break
        chat.say(
            f"That folder doesn't look like a Flow dataset: missing velocity/ under {source}. Try again."
        )

    default_flow = source.name
    flow_choices = _import_flow_targets()
    with prompt_gate():
        if flow_choices:
            flow = questionary.autocomplete(
                "Flow name (copied under data/<flow-name>/)",
                choices=flow_choices,
                default=default_flow,
            ).ask()
        else:
            flow = questionary.text(
                "Flow name (copied under data/<flow-name>/)",
                default=default_flow,
            ).ask()

        overwrite = questionary.confirm(
            "Overwrite existing data/<flow-name>/ if it already exists?",
            default=False,
        ).ask()

    if not flow or not flow.strip():
        raise KeyboardInterrupt

    return LaunchOptions(
        command="import",
        import_data_source=source,
        import_data_flow=flow.strip(),
        import_data_overwrite=bool(overwrite),
    )


def _visualize_options(chat: FlowChat) -> LaunchOptions:
    input_path = _ask_path(
        chat,
        "Path to the trajectory bundle (.npz/.npy)",
        required=True,
        default_key="trajectory_bundle",
    )
    max_particles = _ask_int(chat, "Maximum number of particles to render", default=200)
    output_path = _ask_path(chat, "Optional path to save the visualized plot")
    with prompt_gate():
        flow_overlay = questionary.confirm(
            "Overlay the velocity flow field?", default=True
        ).ask()

    assert max_particles is not None, "I cannot proceed without a particle count."

    return LaunchOptions(
        command="visualize",
        input_path=input_path,
        max_particles=max_particles,
        output_path=output_path,
        flow_overlay=flow_overlay,
    )


def _modeling_options(chat: FlowChat) -> LaunchOptions:
    input_path = _ask_path(
        chat,
        "Path to the trajectory bundle (.npz/.npy)",
        required=True,
        default_key="trajectory_bundle",
    )
    latent_dim = _ask_int(chat, "Latent dimension size", default=16)
    context_dim = _ask_int(chat, "Context dimensionality", default=64)
    encoder_steps = _ask_int(chat, "Encoder training iterations", default=8)
    encoder_lr = _ask_float(chat, "Encoder learning rate", default=0.001)
    cnf_steps = _ask_int(chat, "CNF training iterations", default=6)
    cnf_lr = _ask_float(chat, "CNF learning rate", default=0.0002)
    cnf_hidden_dim = _ask_int(chat, "CNF hidden units", default=128)
    with prompt_gate():
        run_tag = (
            questionary.text("Optional run tag for checkpoints", default="").ask() or ""
        )

    assert latent_dim is not None, "Latent dimension is required."
    assert context_dim is not None, "Context dimensionality is required."
    assert encoder_steps is not None, "Encoder step count is required."
    assert cnf_steps is not None, "CNF step count is required."
    assert cnf_hidden_dim is not None, "CNF hidden size is required."

    return LaunchOptions(
        command="model",
        model_input_path=input_path,
        model_run_name=run_tag.strip() or None,
        encoder_latent_dim=int(latent_dim),
        encoder_context_dim=int(context_dim),
        encoder_steps=int(encoder_steps),
        encoder_lr=float(encoder_lr),
        cnf_steps=int(cnf_steps),
        cnf_lr=float(cnf_lr),
        cnf_hidden_dim=int(cnf_hidden_dim),
    )


def _import_model_targets() -> list[str]:
    """Return candidate folder names under cache/checkpoints.

    We derive defaults from:
    - `cache/checkpoints/<name>` folders already present locally.
    - Any `cache/checkpoints/...` occurrences inside `src/experiments/*.yaml`.
    """

    targets: set[str] = set()

    def collect(obj: object) -> None:
        if isinstance(obj, dict):
            for value in obj.values():
                collect(value)
        elif isinstance(obj, list):
            for value in obj:
                collect(value)
        elif isinstance(obj, str) and "cache/checkpoints/" in obj:
            marker = "cache/checkpoints/"
            tail = obj.split(marker, 1)[1]
            folder = tail.split("/", 1)[0].strip()
            if folder:
                targets.add(folder)

    experiments_dir = Path("src/experiments")
    if experiments_dir.exists():
        for experiment in sorted(experiments_dir.glob("*.yaml")):
            try:
                collect(yaml.safe_load(experiment.read_text()))
            except Exception:
                continue

    ckpt_root = Path("cache/checkpoints")
    if ckpt_root.exists():
        for child in ckpt_root.iterdir():
            if child.is_dir():
                targets.add(child.name)

    return sorted(targets)


def _import_model_options(chat: FlowChat) -> LaunchOptions:
    source_path = _ask_path(
        chat,
        "Path to a downloaded checkpoint folder OR a single .pt file.",
        required=True,
        default_key="import_model_source",
    )
    targets = _import_model_targets()

    with prompt_gate():
        import_bundle = questionary.confirm(
            "Does this folder contain MULTIPLE checkpoint folders (e.g. 2d-euler-vortex_cnf, 2d-euler-vortex_vsde, ...)?",
            default=True,
        ).ask()

    target: str | None = None
    if not import_bundle:
        with prompt_gate():
            if targets:
                target = questionary.select(
                    "Target folder under cache/checkpoints/", choices=targets
                ).ask()
            else:
                target = questionary.text(
                    "Model name (folder under cache/checkpoints)", default="my-model"
                ).ask()
        assert target, "Target folder name is required."

    with prompt_gate():
        overwrite = questionary.confirm(
            "Overwrite existing checkpoint folders if they already exist?",
            default=False,
        ).ask()

    return LaunchOptions(
        command="import-model",
        import_model_source=source_path,
        import_model_target=str(target).strip() if target else None,
        import_model_overwrite=bool(overwrite),
    )


def _ask_int(
    chat: FlowChat,
    prompt: str,
    *,
    default: int | None = None,
    allow_blank: bool = False,
) -> int | None:
    default_text = str(default) if default is not None else ""
    while True:
        with prompt_gate():
            response = questionary.text(prompt, default=default_text).ask()
        if response is None:
            raise KeyboardInterrupt
        candidate = response.strip()
        if not candidate:
            if default is not None and not allow_blank:
                return default
            if allow_blank:
                return None
            chat.say("Please enter a number before continuing.")
            continue
        try:
            return int(candidate)
        except ValueError:
            chat.say("That wasn't a whole number. Please try again.")


def _ask_float(chat: FlowChat, prompt: str, *, default: float) -> float:
    default_text = str(default)
    while True:
        with prompt_gate():
            response = questionary.text(prompt, default=default_text).ask()
        if response is None:
            raise KeyboardInterrupt
        candidate = response.strip()
        if not candidate:
            return default
        try:
            return float(candidate)
        except ValueError:
            chat.say("Please enter a valid number (decimal allowed).")


def _ask_path(
    chat: FlowChat,
    prompt: str,
    *,
    required: bool = False,
    default_key: str | None = None,
) -> Path | None:
    default_value = chat.default_path(default_key) if default_key else None
    default_text = default_value.as_posix() if default_value else ""
    while True:
        with prompt_gate():
            response = questionary.text(prompt, default=default_text).ask()
        if response is None:
            raise KeyboardInterrupt
        candidate = response.strip()
        if not candidate:
            if default_value:
                chat.say(f"Using the last known path: {default_value}")
                if default_key:
                    chat.remember_path(default_key, default_value)
                return default_value
            if required:
                chat.say("This path is required. Please try again.")
                continue
            return None
        resolved = Path(candidate).expanduser()
        if default_key:
            chat.remember_path(default_key, resolved)
        return resolved


def _handle_experiment_cli(args: argparse.Namespace, progress_mode: str | None) -> bool:
    if not (args.list_experiments or args.run_experiment):
        return False
    experiments_dir = (
        Path(args.experiments_dir).expanduser() if args.experiments_dir else None
    )
    orchestrator = ExperimentOrchestrator(
        experiments_dir=experiments_dir,
        progress_mode=progress_mode,
        step_progress_mode="plain",
    )
    if args.list_experiments:
        specs = orchestrator.list_experiments()
        if not specs:
            print("No experiments found.")
        else:
            for spec in specs:
                print(f"- {spec.slug}: {spec.name} — {spec.description}")
        if not args.run_experiment:
            return True
    if args.run_experiment:
        orchestrator.run(args.run_experiment)
    return True


def _run_experiment_menu(chat: FlowChat, progress_mode: str | None) -> None:
    orchestrator = ExperimentOrchestrator(
        console=chat.console,
        progress_mode=progress_mode,
        step_progress_mode="plain",
    )
    experiments = orchestrator.list_experiments()
    if not experiments:
        chat.say("No experiment definitions found under src/experiments yet.")
        return

    choices = [
        questionary.Choice(title=f"{spec.name} — {spec.description}", value=spec.slug)
        for spec in experiments
    ]
    choices.append(questionary.Choice(title="Back", value=None))
    with prompt_gate():
        selection = questionary.select(
            "Select an experiment to run", choices=choices
        ).ask()
    # Some terminals / questionary backends may return the title string for the
    # "Back" choice even when its explicit value was set to None. Guard against
    # that by treating any falsy selection or the literal title as a cancel.
    if not selection or selection == "Back":
        return
    chat.say(
        f"Launching experiment '{selection}'. Live logs will appear below; previous menu saved for later."
    )
    chat.console.print("\n" * 2)
    orchestrator.run(selection)
    chat.console.print(
        "\n[dim]Returning to the experiment menu. You can select another run or exit.[/]\n"
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    preferred_progress = _configure_progress_mode(args.progress_mode)
    if _handle_experiment_cli(args, preferred_progress):
        return

    config = load_config()
    if not args.skip_banner:
        print(welcome_message(config.version))
        print("PIVO experiment CLI — type --help for automation options.\n")
        input("Press Enter to continue...")

    run_conversational_cli(
        config, dev_mode=args.dev_mode, progress_mode=preferred_progress
    )


def _configure_progress_mode(choice: str | None) -> str | None:
    selected = choice or "plain"
    if selected == "auto":
        return os.environ.get("FLOW_PROGRESS_MODE")
    os.environ["FLOW_PROGRESS_MODE"] = selected
    return selected
