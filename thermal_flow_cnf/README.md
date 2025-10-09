# thermal_flow_cnf

A modular Python project for simulating 2D stochastic particle trajectories in fluid flows and learning their distributions with a Continuous Normalizing Flow (CNF).

## Features

- Overdamped Langevin simulator with reflecting boundaries at y = ±H
- Canonical flow profiles: uniform, Couette, and Poiseuille
- CNF model using torchdiffeq.odeint_adjoint with conditional inputs [x0, θ]
- Training, evaluation metrics (MSD, KL, overlap), and visualization utilities
- CLI via `main.py` for simulate/train/evaluate

## Quickstart

1. Install requirements

   ```bash
   pip install -r thermal_flow_cnf/requirements.txt
   ```

2. Run a quick simulation

   ```bash
   python thermal_flow_cnf/main.py simulate --flow poiseuille --num 100
   ```

3. Train the CNF

   ```bash
   python thermal_flow_cnf/main.py train --epochs 5 --batch 128
   ```

4. Evaluate

   ```bash
   python thermal_flow_cnf/main.py evaluate --metric msd
   ```

See `notebooks/` for exploratory examples.

