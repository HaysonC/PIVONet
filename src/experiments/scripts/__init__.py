"""Helper scripts that experiments can invoke as pipeline steps.

===================================================================================
OVERVIEW
===================================================================================
Scripts in this directory are Python modules designed to be called as:
    python -m src.experiments.scripts.{script_name}

They implement specific training routines and experiment-specific logic not
suitable for the generic workflows/ directory.

Each script:
    - Has its own argparse interface
    - Is reproducible (seeds, no random behavior)
    - Logs metrics for pipeline consumption
    - Handles checkpointing and resumption

===================================================================================
SCRIPTS
===================================================================================

train_cnf.py:
    Purpose: Experiment-specific CNF training with detailed logging
    Imports dataset from YAML spec, applies experiment hyperparameters
    Saves intermediate checkpoints and metrics logs
    
    Entry: python -m src.experiments.scripts.train_cnf
    
    Args (from YAML or CLI):
      --experiment: str (e.g., "euler-vortex")
      --config: path to experiment YAML
      --resume: bool (resume from checkpoint if exists)

train_variational_sde.py:
    Purpose: Experiment-specific Variational SDE training
    Applies β-VAE annealing schedule
    Tracks KL divergence and reconstruction loss separately
    
    Entry: python -m src.experiments.scripts.train_variational_sde

===================================================================================
INTEGRATION WITH ORCHESTRATOR
===================================================================================

YAML example:

    steps:
      - name: "Train CNF"
        script: "src/experiments.scripts.train_cnf"
        params:
          experiment: "euler-vortex"
          epochs: 100
          hidden_dim: 128

Orchestrator converts params to CLI:
    python -m src.experiments.scripts.train_cnf \\
        --experiment euler-vortex \\
        --epochs 100 \\
        --hidden_dim 128

===================================================================================
"""

"""Helper scripts that experiments can invoke."""
