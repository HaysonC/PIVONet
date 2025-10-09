# Thermal Flow CNF

Model, simulate, and analyze thermal particle transport in simple channel flows using a Continuous Normalizing Flow (CNF). This project provides:

- A simulator for overdamped Langevin dynamics with reflecting boundaries and configurable flow profiles (uniform, Couette, Poiseuille)
- A CNF model (Neural ODE) trained to model final particle positions conditioned on initial state/context
- A Streamlit app with an end-to-end workflow: simulate → train → animate → analyze metrics
- Apple Silicon (MPS) support and robust animation embedding controls


## Features

- Simulation
  - Overdamped Langevin trajectories with reflecting y-boundaries at ±H
  - Built-in flows: uniform, Couette, and Poiseuille
  - Initial distribution options:
    - Uniform: x∈[0,1], y∈[−H, H]
    - Gaussian: user-specified mean and covariance (saved as metadata with the dataset)
- CNF Model
  - Neural ODE with torchdiffeq on CUDA/CPU; on Apple MPS a fixed-step RK4 fallback keeps everything float32-safe
  - Log-likelihood training with AdamW, cosine LR schedule, grad clipping
  - Checkpoints include shape metadata (class/dim/cond_dim/hidden_dim) for compatibility
- Visualization and UI
  - Streamlit app with tabs: Data, Train, Inference & Animate, Metrics & Analysis, Logs
  - Flow quiver overlay, trajectories, and animation with an initial Gaussian ellipse and mean marker
  - Animation size controls: max frames, stride (downsampling), and embed size limit (MB)
  - Checkpoint selection is optional; the app will run with a fresh model if none or incompatible
  - Live progress, logs, and toasts for long-running steps


## Project layout

```text
. 
├── run_app.sh                         # Launches the Streamlit app
├── requirements.txt                   # Python dependencies
├── thermal_flow_cnf/
│   ├── app_streamlit.py              # Streamlit UI (simulate/train/animate/metrics/logs)
│   ├── main.py                       # CLI entry points (simulate/train/evaluate)
│   └── src/
│       ├── config.py
│       ├── flows/                    # Uniform, Couette, Poiseuille
│       ├── simulation/
│       │   ├── langevin.py           # Simulate trajectories + dataset generation
│       │   └── dataset.py            # PyTorch Dataset wrapper
│       ├── model/
│       │   ├── net.py                # MLP backbone
│       │   ├── base_cnf.py           # ODEFunc + CNF, MPS-safe integration
│       │   └── train.py              # Training loop with progress callback
│       ├── evaluation/
│       │   ├── visualize.py          # Plots + animations (with size controls)
│       │   └── metrics.py            # MSD, KL, overlap
│       └── utils/
│           └── io.py                 # Checkpoint save/load with metadata
└── notebooks/                        # Optional exploratory notebooks
```


## Quickstart

### 1) Environment

You’ll need Python 3.10+ (3.11 recommended). Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you’re on Apple Silicon and want to use MPS, use a recent PyTorch build that supports MPS. The app also supports CPU and CUDA.

### 2) Run the Streamlit app

Use the provided script (it sets PYTHONPATH for local imports):

```bash
source run_app.sh
```

This launches a browser at a local URL. The app is organized into tabs:

- Data: Simulate datasets
  - Choose flow (uniform/Couette/Poiseuille), simulation parameters (N, T, dt, D, H)
  - Choose the initial distribution (uniform or Gaussian) and, if Gaussian, specify mean and covariance
  - Simulate and select a dataset for plotting; MSD and a static trajectory plot are shown
- Train: Train the CNF
  - Choose device (cuda/mps/cpu), hidden size, epochs, and batch size
  - Training progress shows NLL, avg logp, and bits-per-dim (bpd)
  - Checkpoints are optional; newest is auto-selected but you can choose “<none>” to run with a fresh model
- Inference & Animate: Animate true vs model trajectories
  - Uses the selected dataset and (optionally) a checkpoint
  - Initial Gaussian ellipse and mean marker are overlaid for context
  - Animation controls:
    - Max frames: caps total frames to keep embedded HTML small
    - Frame stride (0 = auto): downsample frames manually if needed
    - Embed limit (MB): allow larger-than-default animations if desired
- Metrics & Analysis: Quick inspection
  - 2D histograms for final positions and simple metrics (KL divergence, overlap ratio)
- Logs: Centralized logs with a clear button

### 3) Device selection

In the sidebar, the app detects available devices: cuda, mps, cpu. Pick the one you want. The code keeps tensors float32-safe on MPS and uses a pure PyTorch RK4 integrator on MPS to avoid float64 issues in adjoint solvers.


## Command-line interface (optional)

You can also simulate/train/evaluate from the CLI. Ensure PYTHONPATH includes the project root (the app script already does this):

```bash
export PYTHONPATH=.
# Simulate
python thermal_flow_cnf/main.py simulate --flow poiseuille --num 1000

# Train
python thermal_flow_cnf/main.py train --dataset thermal_flow_cnf/data/raw/<your_dataset>.npz --epochs 20 --batch 256

# Evaluate (MSD)
python thermal_flow_cnf/main.py evaluate --dataset thermal_flow_cnf/data/raw/<your_dataset>.npz --metric msd
```

Generated artifacts:

- Datasets: `thermal_flow_cnf/data/raw/*.npz` (contains trajs, x0s, thetas, dt, D, H, plus init_mean/init_cov)
- Checkpoints: `thermal_flow_cnf/checkpoints/*.pt` (with metadata for compatibility)


## Troubleshooting

- Apple Silicon (MPS) float64 error
  - The app forces float32 on MPS for all ODE states and time grids
  - On MPS it uses a fixed-step RK4 integrator instead of the adjoint method to avoid dtype pitfalls
  - You can always switch to CPU in the sidebar if you prefer

- Animation too large for embedding (20 MB limit)
  - Use “Max frames” and “Frame stride” to downsample frames
  - Increase “Embed limit (MB)” if you really want a larger inline animation (may impact page responsiveness)

- Checkpoint incompatibility
  - Checkpoints are validated against model metadata; if incompatible, the app warns and runs with a fresh model
  - You can select "<none>" to skip loading any checkpoint

- Model loss becomes negative
  - The optimization target is NLL = −log p(x|context). As the model improves, log-likelihood increases and NLL can become negative—this is normal for densities
  - The UI displays NLL, avg logp, and bpd to make interpretation clear


## Notes

- This project is educational/research-oriented, focusing on clarity and a smooth UI for experimentation
- If you enable very long trajectories or large particle counts, prefer downsampling animations and running training on CUDA


## License

No explicit license specified. If you plan to use or distribute, consider adding a license file.
 
