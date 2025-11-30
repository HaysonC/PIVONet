# Flow Toolchain Usage Guide

This document walks through the supported interfaces of the Flow project and shares tips for faster end-to-end runs.

## Core Entry Point

- Install the package (and expose the `pivo` command) with `python -m pip install -e .` during development or `python -m pip install .` for a standard install.
- Run `pivo` from any shell to open the conversational CLI immediately.
- Run `source venv/bin/activate` and then `./main.sh` to run the preflight checks, install requirements, and launch the Python entry point.
- The shell script now leaves you inside your default shell if preflight questions fail so you can inspect the message or rerun commands.
- `python -m src.main` prints a welcome banner and asks whether you want the conversation-style CLI or the Streamlit GUI.
- To launch YAML-driven experiments, run `pivo` and pick **experiment**, or call `pivo --list-experiments` / `pivo --run-experiment <slug>` directly. YAML files live in `src/experiments/` alongside any helper scripts.

## Conversational CLI

- Commands: `import`, `visualize`, `velocity`, and `model`.
- Use the interactive prompts to adjust particle counts, sampling steps, or encoder/CNF training hyperparameters. Defaults come from `config.yml`.
- After every command, Flow prints a recap and command hint (`flow-cli <command> ...`) plus this guide path so you can script a repeatable invocation.
- Errors show a friendly message and a pointer to this guide.
- When a prompt asks for a bundle or velocity path you just processed, leave the question blank and Flow will reuse the last result automatically.

## Streamlit GUI

- Spin up `streamlit run src/app/ui.py` or choose the "Graphical (GUI)" option from the launcher.
- The sidebar lets you select one of the four modes and presents forms with the same hyperparameters as the CLI.
- Each form validates inputs, runs the shared command handlers, and reports the generated artifacts plus a link back to this documentation.

## Command Descriptions

1. **Import**: ingest PyFR-generated velocity snapshots, simulate particles, and save trajectories under `data/cfd/trajectories/`.
2. **Visualize**: plot an exported trajectory bundle with optional velocity overlay.
3. **Velocity**: sample a raw velocity `.npy` snapshot and draw an arrow field.
4. **Model**: fit the diffusion encoder and CNF on a saved trajectory bundle. The resulting checkpoints land in `cache/modeling/<run_tag>/` and you can rerun the same configuration by copying the hint line.
5. **Experiment**: choose a YAML pipeline (e.g., `demo-baseline`) and let the orchestrator run each scripted step with labeled device-aware progress bars.

## Helpful Tips

- The CLI/GUI share the same `LaunchOptions` definition (see `src/interfaces/launch_options.py`), meaning any hyperparameter you tune in one interface can be replicated in the other.
- Use the FlowChat hint (or Streamlit caption) to grab a fully expanded command you can add to scripts or CI.
- When training the hybrid model, feed the absolute path to your `.npz` bundle and watch the printed metrics; the CLI prints encoder/CNF losses, while the GUI surfaces them as Streamlit `metric` badges.
- Pipelines are regular YAML files—copy `src/experiments/demo-baseline.yaml` to add new experiments, point each step at scripts or commands you need, and the orchestrator will handle ordering and progress UI for you.

Refer here whenever you need a refresher—Flow will remind you of this guide after each run so automation is painless.
