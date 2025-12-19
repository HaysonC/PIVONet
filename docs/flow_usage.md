# Flow Toolchain Usage Guide

This document walks through the supported interfaces of the Flow project and shares tips for faster end-to-end runs.

## Core Entry Point

> [!WARNING]
> The GUI mode is experimental and not recommended.
> When you run `pivo`, you will be prompted to choose **CLI** or **GUI** — choose **CLI**.

- Install the package (and expose the `pivo` command) with `python -m pip install -e .` during development or `python -m pip install .` for a standard install.
- Run `pivo` from any shell to open the conversational CLI immediately.
- Run `source venv/bin/activate` and then `./main.sh` to run the preflight checks, install requirements, and launch the Python entry point.
- The shell script now leaves you inside your default shell if preflight questions fail so you can inspect the message or rerun commands.

> [!IMPORTANT]
> To run experiments, just run `pivo` and pick **experiment** in the conversational menu.
> That is the intended path (no extra flags needed).

> [!NOTE]
> Pretrained checkpoints and example data are available here:
> https://drive.google.com/drive/folders/13ykGleAmZTNAz1lhR0x6FekYqyGYbper?usp=sharing

## Conversational CLI

- Commands: `import`, `visualize`, `model`, and `experiment`.
- Use the interactive prompts to adjust particle counts, sampling steps, or encoder/CNF training hyperparameters. Defaults come from `config.yml`.
- After every command, Flow prints a recap and command hint (`flow-cli <command> ...`) plus this guide path so you can script a repeatable invocation.
- Errors show a friendly message and a pointer to this guide.
- When a prompt asks for a bundle path you just processed, leave the question blank and Flow will reuse the last result automatically.

> [!IMPORTANT]
> The `import` menu has two sub-options:
> - **Flow data**: copy a downloaded dataset folder (must include `velocity/`) into `data/<flow-name>/`.
> - **Model checkpoints (pretrained)**: copy downloaded checkpoint folders into `cache/checkpoints/`.

## Command Descriptions

1. **Import**:
	- **Flow data**: copy a downloaded dataset folder (must include `velocity/`) into `data/<flow-name>/` so experiments can run.
	- **Model checkpoints (pretrained)**: import a downloaded checkpoint bundle so you can run inference without retraining.
2. **Visualize**: plot an exported trajectory bundle with optional velocity overlay.
3. **Model**: fit the diffusion encoder and CNF on a saved trajectory bundle. The resulting checkpoints land in `cache/modeling/<run_tag>/` and you can rerun the same configuration by copying the hint line.
4. **Experiment**: choose a YAML pipeline (e.g., `2d-euler-vortex`) and let the orchestrator run each scripted step with labeled device-aware progress bars.

> [!TIP]
> Pretrained inference workflow:
> 1) Download checkpoints from the Google Drive link.
> 2) Run `pivo` → `import` → `Model checkpoints (pretrained)` and select the downloaded folder.
> 3) Run `pivo` → `experiment` and select the corresponding dataset.
> 4) When training steps detect existing checkpoints, choose **Skip step (reuse cached checkpoint)**.

## Helpful Tips

- Use the FlowChat hint to grab a fully expanded command you can add to scripts or CI.
- When training the hybrid model, feed the absolute path to your `.npz` bundle and watch the printed metrics; the CLI prints encoder/CNF losses.
- Pipelines are regular YAML files—copy `src/experiments/demo-baseline.yaml` to add new experiments, point each step at scripts or commands you need, and the orchestrator will handle ordering and progress UI for you.

Refer here whenever you need a refresher—Flow will remind you of this guide after each run so automation is painless.

## VSDE Inference Integrator Comparison

Pass `--integrator` (`euler`, `improved_euler`, `rk4`, or `dopri5`) to `src/workflows/run_vsde_inference.py` to swap the deterministic drift solver while diffusion noise still uses Euler–Maruyama. Use `pivo` → `experiment` → `integrator-comparison` to run the sweep and inspect overlays under `cache/artifacts/integrator-comparison/<method>`.

The experiment now ends with a `compare_integrators.py` helper that compiles the MAE metrics across the runs and writes `comparison_mae.png` plus `comparison_summary.json` under `cache/artifacts/integrator-comparison/charts/`.

````
