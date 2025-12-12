"""Experiment pipeline definitions and helper scripts.

===================================================================================
OVERVIEW
===================================================================================
This package contains:
  - YAML experiment specifications (declarative pipelines)
  - Python helper scripts (individual workflow steps)
  - Documentation for running predefined experiments

Main entry: pivo --run-experiment <slug> or ExperimentOrchestrator.run()

===================================================================================
EXPERIMENT SPECIFICATIONS
===================================================================================

Location: src/experiments/*.yaml

Structure:
    name: "Demo Baseline"
    slug: "demo-baseline"
    description: "Sanity check: simulate, train CNF, evaluate"
    
    steps:
      - name: "Import Trajectories"
        script: "src/workflows/import_trajectories.py"
        params:
          particles: 100
          max_steps: 50
          dt: 0.01
          output_dir: "data/demo"
      
      - name: "Train CNF"
        script: "src/workflows/train_cnf.py"
        params:
          trajectory_path: "data/demo/*.npz"
          hidden_dim: 64
          epochs: 50

Each step is standalone Python script with:
  - argparse for CLI arguments
  - Logging to stdout/stderr
  - Exit code 0 on success, non-zero on error

===================================================================================
HELPER SCRIPTS
===================================================================================

scripts/ subdirectory:

train_cnf.py:
    Entry: python -m src.experiments.scripts.train_cnf
    Trains a CNF model on trajectory bundle
    Saves checkpoint to cache/checkpoints/

train_variational_sde.py:
    Entry: python -m src.experiments.scripts.train_variational_sde
    Trains a variational SDE on trajectory bundle
    Saves checkpoint and encoder weights

===================================================================================
BUILT-IN EXPERIMENTS
===================================================================================

1. demo-baseline: Full pipeline on small synthetic data
   - Simulate particles in demo velocity field
   - Train CNF for 50 epochs
   - Generate metrics report

2. euler-vortex: Complete workflow on 2D Euler vortex
   - Simulate 1000 particles through vortex dynamics
   - Train CNF + SDE models
   - Render trajectory animations
   - Evaluate path length and overlap metrics

3. viscous-shock-tube: 2D viscous shock tube flow
   - Simulate diffusion-dominated flow
   - Train variational SDE
   - Compare CNF vs SDE predictions

4. inc-cylinder: Incompressible cylinder flow
   - CFD data from PyFR (if available)
   - Train on real CFD trajectories
   - Benchmark model accuracy

5. velocity-animations: Render quiver animations for all flows
   - Generates GIF/MP4 for every velocity dataset
   - Configurable stride, frame rate, resolution

===================================================================================
RUNNING EXPERIMENTS
===================================================================================

# Interactive selection
pivo

# List all experiments
pivo --list-experiments

# Run specific experiment
pivo --run-experiment demo-baseline

# Run with parameter overrides
pivo --run-experiment demo-baseline --overrides particles=500 epochs=100

# Progress tracking
    - Console output from each step
    - Rich progress bar (overall pipeline)
    - Elapsed time and ETA display
    - Save summary report to cache/

===================================================================================
ADDING NEW EXPERIMENTS
===================================================================================

1. Create YAML file: src/experiments/my_experiment.yaml
   
    name: "My Experiment"
    slug: "my-experiment"
    description: "Brief description"
    steps:
      - name: "Step 1"
        script: "path/to/script.py"
        params: {...}

2. Create associated helper scripts (if needed) under:
   - src/workflows/
   - src/experiments/scripts/

3. Verify YAML syntax and step scripts are executable

4. Test: pivo --run-experiment my-experiment

===================================================================================
"""

"""Experiment pipeline definitions and helper scripts."""
