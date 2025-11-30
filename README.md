# PIVONet (Flow)

Physics-Informed Variational ODE Networks (PIVONet) fuses PyFR-based CFD data ingestion with hybrid diffusion/CNF modeling and dual interfaces (conversational CLI + Streamlit GUI).

## Installation

Use any modern Python (>= 3.10). The project ships with a `pyproject.toml`, so you can install it directly:

```bash
python -m pip install .
```

For development, install in editable mode so local changes are reflected immediately:

```bash
python -m pip install -e .
```

Both commands respect `requirements.txt`, so you get the same dependency set used in the repo.

## Usage

### Conversational CLI (`pivo`)

After installation a console script named `pivo` becomes available. It launches the conversational Rich/Questionary CLI:

```bash
pivo
```

The CLI lets you import velocity snapshots, visualize saved bundles, sketch velocity fields, or train the hybrid diffusion+CNF model. When a prompt asks for a previously produced trajectory or velocity file, you can leave it blank to reuse the last successful path.

#### YAML Experiment Pipelines

Select **experiment** from the CLI menu (or run `pivo --list-experiments` / `pivo --run-experiment <slug>`) to inspect YAML pipelines stored under `src/experiments/`. Each YAML file defines metadata plus a list of steps that invoke helper scripts. When you choose a pipeline, the orchestrator detects your compute device, prints `[m/n]` progress labels, and shows a Rich loading bar while each script runs. Try the included `demo-baseline` pipeline for a quick sanity check (`pivo --run-experiment demo-baseline`).

### Launcher / GUI

If you prefer the original launcher that lets you choose between CLI and Streamlit GUI, run:

```bash
python -m src.main
```

Or go straight to the GUI once dependencies are installed:

```bash
streamlit run src/app/ui.py
```

## Project Layout

- `src/` – Core Python package (CLI, CFD utilities, hybrid models, visualization helpers).
- `docs/flow_usage.md` – Detailed walkthrough of both interfaces and automation tips.
- `src/experiments/` – YAML experiment definitions and helper scripts.
- `config.yml` – Simulation defaults shared by CLI/GUI.
- `requirements.txt` – Exact dependency pins reused by the packaging metadata.

## License

Released under the MIT License (see `LICENSE`).
