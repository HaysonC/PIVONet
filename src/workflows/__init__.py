"""Reusable workflow scripts callable from the PIVO orchestrator or CLI.

===================================================================================
OVERVIEW
===================================================================================
This package contains standalone Python scripts that implement individual
workflow steps. Each script:
  - Accepts command-line arguments (--help for usage)
  - Runs independently (no GUI required)
  - Returns exit code 0 on success
  - Logs progress to stdout/stderr

Scripts are invoked by:
  1. ExperimentOrchestrator (via YAML pipeline definitions)
  2. Direct CLI: python -m src.workflows.script_name
  3. User code: subprocess.run(["python", "-m", "src.workflows...."]) 

===================================================================================
WORKFLOW SCRIPTS
===================================================================================

import_trajectories.py:
    Purpose: Simulate particle trajectories from velocity field
    Entry: python -m src.workflows.import_trajectories
    
    Args:
      --velocity-dir PATH        Velocity snapshot directory (default: config)
      --particles INT            Number of particles (default: config)
      --max-steps INT            Max integration steps (default: config)
      --dt FLOAT                 Time step (default: config)
      --output-dir PATH          Save .npz bundle here (default: data/cfd)
    
    Output: trajectories_n{particles}_{timestamp}.npz
    
    Example:
      python -m src.workflows.import_trajectories \
        --velocity-dir data/2d-euler-vortex/velocity \
        --particles 500 \
        --max-steps 100 \
        --dt 0.01

train_cnf.py:
    Purpose: Train Continuous Normalizing Flow on trajectory bundle
    Entry: python -m src.workflows.train_cnf
    
    Args:
      --trajectory PATH          Path to .npz trajectory bundle
      --hidden-dim INT           Network width (default: 128)
      --depth INT                Network depth (default: 3)
      --lr FLOAT                 Learning rate (default: 1e-3)
      --epochs INT               Training epochs (default: 100)
      --batch-size INT           Batch size (default: 32)
      --output-dir PATH          Save checkpoint here
    
    Output: cnf_latest.pt, cnf_best.pt, training_log.json
    
    Example:
      python -m src.workflows.train_cnf \
        --trajectory data/demo/trajectories.npz \
        --epochs 50

train_variational_sde.py:
    Purpose: Train Variational SDE model
    Similar to train_cnf.py but outputs: vsde_latest.pt, encoder.pt

render_velocity_animations.py:
    Purpose: Generate GIF/MP4 animations of velocity fields
    Entry: python -m src.workflows.render_velocity_animations
    
    Args:
      --velocity-dir PATH        Single flow velocity directory
      --velocity-root PATH       Or scan entire root for multiple flows
      --output-dir PATH          Save animations here (default: cache/artifacts)
      --vector-stride INT        Decimation for quivers (default: 3)
      --fps INT                  Animation frame rate (default: 12)
      --device STR               Device: auto|cuda|mps|cpu (default: auto)
    
    Output: {flow_name}_velocity.gif
    
    Example:
      python -m src.workflows.render_velocity_animations \
        --velocity-dir data/2d-euler-vortex/velocity \
        --output-dir cache/artifacts/

render_velocity_field.py:
    Purpose: Render static quiver plots of velocity field
    Args: Similar to render_velocity_animations
    Output: {flow_name}_velocity_field.png

prepare_demo_artifacts.py:
    Purpose: Generate lightweight synthetic trajectory for demos
    Entry: python -m src.workflows.prepare_demo_artifacts
    
    Args:
      --output PATH              Save .npz here (default: data/demo/demo_bundle.npz)
      --particles INT            Num particles (default: 64)
      --steps INT                Num timesteps (default: 40)
      --dt FLOAT                 Time step (default: 0.05)
    
    Output: Synthetic .npz bundle (useful for testing without real CFD data)

summarize_demo_artifacts.py:
    Purpose: Print statistics of synthetic bundle
    Entry: python -m src.workflows.summarize_demo_artifacts

run_vsde_inference.py:
    Purpose: Generate trajectories using trained VSDE model
    Args: --checkpoint, --particles, --steps, --output-dir
    Output: Sampled trajectories as .npz

===================================================================================
COMMON PATTERNS
===================================================================================

All scripts follow this structure:

    import argparse
    from pathlib import Path
    from typing import Sequence
    
    def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--arg1", type=int, default=100)
        parser.add_argument("--arg2", type=str, required=True)
        return parser.parse_args(argv)
    
    def main(argv: Sequence[str] | None = None) -> None:
        args = _parse_args(argv)
        # Main logic here
        print("Success!")
    
    if __name__ == "__main__":
        main()

Entry via python -m:
    - Automatically calls main() if module executed directly
    - Can also import: from src.workflows.script import main; main()

Exit codes:
    - 0: Success
    - 1: Argument parsing error
    - 2: File not found / permission error
    - 3: Computation error (ODE divergence, NaN loss, etc.)

===================================================================================
ORCHESTRATOR INTEGRATION
===================================================================================

YAML experiments invoke scripts via ExperimentOrchestrator:

    steps:
      - name: "Simulate"
        script: "src/workflows/import_trajectories.py"
        params:
          particles: 100
          velocity_dir: "data/2d-euler-vortex/velocity"

Orchestrator:
    1. Creates unique run directory under cache/
    2. cd to project_root()
    3. Invokes: python -m {script} --{param_key} {param_value} ...
    4. Captures stdout/stderr
    5. Checks exit code
    6. If success: proceed to next step
       If failure: abort pipeline, report error

===================================================================================
DEPENDENCIES
===================================================================================

All scripts depend on:
    - src.utils (paths, config, I/O)
    - src.interfaces (data structures)
    - numpy, torch, scipy

Optional:
    - PyFR (for CFD simulation workflows)
    - Taichi (for GPU-accelerated visualization)
    - matplotlib (for static plots)
    - ffmpeg (for MP4 encoding)

Install via: pip install -r requirements.txt

===================================================================================
"""

"""Reusable workflow scripts runnable from the PIVO orchestrator or CLI."""
