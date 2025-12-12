# Flow Toolchain Usage Guide

This document walks through the supported interfaces of the Flow project and shares tips for faster end-to-end runs.

## Core Entry Point

- Install the package (and expose the `pivo` command) with `python -m pip install -e .` during development or `python -m pip install .` for a standard install.
- Run `pivo` from any shell to open the conversational CLI immediately.
- Run `source venv/bin/activate` and then `./main.sh` to run the preflight checks, install requirements, and launch the Python entry point.
- The shell script now leaves you inside your default shell if preflight questions fail so you can inspect the message or rerun commands.
- To launch YAML-driven experiments, run `pivo` and pick **experiment**, or call `pivo --list-experiments` / `pivo --run-experiment <slug>` directly. YAML files live in `src/experiments/` alongside any helper scripts.

## Conversational CLI

- Commands: `import`, `visualize`, `velocity`, and `model`.
- Use the interactive prompts to adjust particle counts, sampling steps, or encoder/CNF training hyperparameters. Defaults come from `config.yml`.
- After every command, Flow prints a recap and command hint (`flow-cli <command> ...`) plus this guide path so you can script a repeatable invocation.
- Errors show a friendly message and a pointer to this guide.
- When a prompt asks for a bundle or velocity path you just processed, leave the question blank and Flow will reuse the last result automatically.

## Command Descriptions

1. **Import**: ingest PyFR-generated velocity snapshots, simulate particles, and save trajectories under `data/cfd/trajectories/`.
2. **Visualize**: plot an exported trajectory bundle with optional velocity overlay.
3. **Velocity**: sample a raw velocity `.npy` snapshot and draw an arrow field.
4. **Model**: fit the diffusion encoder and CNF on a saved trajectory bundle. The resulting checkpoints land in `cache/modeling/<run_tag>/` and you can rerun the same configuration by copying the hint line.
5. **Experiment**: choose a YAML pipeline (e.g., `demo-baseline`) and let the orchestrator run each scripted step with labeled device-aware progress bars.

## Helpful Tips

- Use the FlowChat hint to grab a fully expanded command you can add to scripts or CI.
- When training the hybrid model, feed the absolute path to your `.npz` bundle and watch the printed metrics; the CLI prints encoder/CNF losses.
- Pipelines are regular YAML files—copy `src/experiments/demo-baseline.yaml` to add new experiments, point each step at scripts or commands you need, and the orchestrator will handle ordering and progress UI for you.

Refer here whenever you need a refresher—Flow will remind you of this guide after each run so automation is painless.

## Velocity Animation Workflow

Need a quick QA pass on the CFD velocity fields themselves? Run the Rich-enabled workflow helper to build quiver animations straight from the `.npy` snapshots:

```bash
python -m src.workflows.render_velocity_animations --velocity-dir data/2d-viscous-shock-tube/velocity --output cache/artifacts/2d-viscous-shock-tube/velocity_evolution.gif --save-preview --device auto
```

- Omit `--velocity-dir` to let the script crawl `--velocity-root` (defaults to `data/`) and render every flow it finds.
- The orchestrator-compatible CLI flags mean you can add the helper as a YAML experiment step (see `velocity-animation` in the packaged pipelines).
- `--device auto` prefers Apple MPS when PyTorch/MPS is present but gracefully falls back to CPU.
- A fully automated pipeline ships as `pivo --run-experiment velocity-animations`, which scans the default data root and writes GIFs + previews to `cache/artifacts/<flow>/`.
