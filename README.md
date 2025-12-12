# PIVONet (Flow)

Physics-Informed Variational ODE Networks (PIVONet) fuses PyFR-based CFD data ingestion with hybrid diffusion/CNF modeling and dual interfaces (conversational CLI + Streamlit GUI).

## Installation

Use strictly Python (= 3.11). The project ships with a `pyproject.toml`, so you can install it directly:

```bash
python -m pip install .
```

For development, install in editable mode so local changes are reflected immediately:

```bash
python -m pip install -e .
```

Both commands respect `requirements.txt`, so you get the same dependency set used in the repo.

**Note:** For the interactive trajectory viewer, install Taichi separately:

```bash
pip install taichi
```

## Usage

### Conversational CLI (`pivo`)

After installation a console script named `pivo` becomes available. It launches the conversational Rich/Questionary CLI:

```bash
pivo
```

The CLI lets you import velocity snapshots, visualize saved bundles, sketch velocity fields, or train the hybrid diffusion+CNF model. When a prompt asks for a previously produced trajectory or velocity file, you can leave it blank to reuse the last successful path.

#### YAML Experiment Pipelines

Select **experiment** from the CLI menu (or run `pivo --list-experiments` / `pivo --run-experiment <slug>`) to inspect YAML pipelines stored under `src/experiments/`. Each YAML file defines metadata plus a list of steps that invoke helper scripts. When you choose a pipeline, the orchestrator detects your compute device, prints `[m/n]` progress labels, and shows a Rich loading bar while each script runs. Try the included `demo-baseline` pipeline for a quick sanity check (`pivo --run-experiment demo-baseline`).

**Note:** To run predefined experiments, ensure that the corresponding density and trajectory data are placed in `data/{experiment}/` directories. These data can be obtained by running the Jupyter notebooks in `src/cfd/jupyter_notebooks/`.

### Interactive Trajectory Viewer

For an interactive 3D visualization of particle trajectories, use the Taichi-based viewer:

```bash
python -m src.visualization.viewer_taichi --dataset 2d-euler-vortex
```

This launches a GPU-accelerated window where you can play/pause trajectories, adjust particle size, toggle colors, and orbit the camera. If a CNF checkpoint is available under `cache/checkpoints/{dataset}_cnf/`, it will be loaded automatically for interactive generation. Alternatively, specify a custom checkpoint path.

```bash
python -m src.visualization.viewer_taichi --dataset 2d-euler-vortex --cnf-checkpoint cache/checkpoints/2d-euler-vortex_cnf/cnf_latest.pt
```

### Velocity Animation Workflow

To render quiver animations directly from CFD velocity snapshots (either for a single flow or by scanning an entire data root), invoke the workflow helper:

```bash
python -m src.workflows.render_velocity_animations --velocity-dir data/2d-euler-vortex/velocity --output cache/artifacts/2d-euler-vortex/velocity_evolution.gif --save-preview --vector-stride 3 --fps 12 --device auto
```

Omit `--velocity-dir` to let the script crawl `--velocity-root` (defaults to `data/`) and produce animations for every detected flow automatically. Pass `--device auto` (default) to prefer Apple MPS when PyTorch/MPS is available, falling back to CPU otherwise.

Prefer YAML pipelines? Run `pivo --run-experiment velocity-animations` to sweep every flow using the orchestrator with Rich progress output.

### VSDE Inference Integrator Comparison

`src/workflows/run_vsde_inference.py` now exposes a `--integrator` flag (choices: `euler`, `improved_euler`, `rk4`, `dopri5`) so you can target different drift solvers while diffusion always uses Euler–Maruyama. Run `pivo --run-experiment integrator-comparison` to execute inference with each integrator back-to-back and inspect the overlays living under `cache/artifacts/integrator-comparison/<method>`.

The final step of the experiment runs `src/experiments/scripts/compare_integrators.py`, which produces `cache/artifacts/integrator-comparison/charts/comparison_mae.png` plus a summary JSON detailing the VSDE/CNF MAE per integrator.

## Project Layout

- `src/` – Core Python package (CLI, CFD utilities, hybrid models, visualization helpers).
- `docs/flow_usage.md` – Detailed walkthrough of both interfaces and automation tips.
- `src/experiments/` – YAML experiment definitions and helper scripts.
- `config.yml` – Simulation defaults shared by CLI/GUI.
- `requirements.txt` – Exact dependency pins reused by the packaging metadata.

## License

Released under the MIT License (see `LICENSE`).
