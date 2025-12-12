"""Training infrastructure: datasets, optimizers, and learning loops.

===================================================================================
OVERVIEW
===================================================================================
Provides PyTorch training utilities for neural ODE models:
  - Dataset classes for trajectory sequences
  - Trainer orchestration with checkpointing
  - Loss functions and metrics
  - Device/dtype management
  - Progress tracking

Main classes: CFDTrajectorySequenceDataset, Trainer

===================================================================================
SUBMODULE STRUCTURE
===================================================================================

datasets.py:
    CFDTrajectorySequenceDataset - PyTorch Dataset wrapper
    Attributes:
      - trajectory_bundles: list[TrajectoryResult]
      - sequence_length: int (num steps to predict forward)
      - normalize: bool
    Methods:
      - __len__() → total samples
      - __getitem__(idx) → (input_traj, target_traj) tensors

trainers.py:
    Trainer - Main training loop orchestrator
    Attributes:
      - model: nn.Module (CNF or SDE)
      - optimizer: torch.optim.Optimizer
      - device: str
      - best_loss: float
      - checkpoint_dir: Path
    Methods:
      - train(dataloader, num_epochs)
      - validate(dataloader)
      - save_checkpoint(path)
      - load_checkpoint(path)

===================================================================================
DATA STRUCTURES
===================================================================================

CFDTrajectorySequenceDataset:
    Input: List of TrajectoryResult objects from simulations
    
    Preprocessing:
      1. Extract (positions, velocities) sequences
      2. Normalize: (x - mean) / std (computed from all data)
      3. Create sliding windows of sequence_length
    
    Output per __getitem__:
      - input_trajectory: (sequence_length, spatial_dim)
      - target_trajectory: (sequence_length, spatial_dim)
      - Both as torch.FloatTensor

Trainer:
    Training loop structure:
      for epoch in range(num_epochs):
          for batch in dataloader:
              optimizer.zero_grad()
              loss = model(batch)
              loss.backward()
              optimizer.step()
              
              if loss < best_loss:
                  save_checkpoint()

===================================================================================
WORKFLOW DIAGRAM
===================================================================================

    TrajectoryResult (CFD simulation)
            │
            ▼
    CFDTrajectorySequenceDataset
        ├─ Normalize
        ├─ Create sequences
        └─ Prepare for training
            │
            ▼
    PyTorch DataLoader
        ├─ Batch size N
        ├─ Shuffle
        └─ num_workers for parallelism
            │
            ▼
    Trainer.train()
        ├─ Forward: model(batch)
        ├─ Loss computation
        ├─ Backward: loss.backward()
        ├─ Optimizer step
        └─ Checkpoint best model
            │
            ▼
    Training complete
        ├─ Best model in checkpoint_dir
        └─ Loss history stored

===================================================================================
USAGE EXAMPLES
===================================================================================

# Prepare dataset
from src.modeling.datasets import CFDTrajectorySequenceDataset
from src.utils.trajectory_io import load_trajectory_bundle
import torch

result = load_trajectory_bundle("output/traj.npz")
dataset = CFDTrajectorySequenceDataset(
    trajectory_bundles=[result],
    sequence_length=10,
    normalize=True
)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2
)

# Train model
from src.networks.cnf import CNFModel
from src.modeling.trainers import Trainer

model = CNFModel(dim=2, hidden_dim=128)
trainer = Trainer(
    model,
    device="mps",
    checkpoint_dir="cache/checkpoints/cnf_run1"
)

trainer.train(
    dataloader,
    num_epochs=100,
    learning_rate=1e-3
)

# Load best checkpoint
trainer.load_checkpoint("cache/checkpoints/cnf_run1/best.pt")

===================================================================================
CONSTRAINTS
===================================================================================

1. Dataset assumes trajectories have consistent spatial dimension
2. Normalize computed per-dataset (statistically independent datasets
   should not share statistics)
3. Trainer expects fixed device for entire training run
4. No multi-GPU support (use DistributedDataParallel externally if needed)
5. Checkpoints include full model state_dict + optimizer state

===================================================================================
MEMORY & PERFORMANCE
===================================================================================

Memory per batch:
    B × L × D × 4 bytes for float32
    B=batch_size, L=sequence_length, D=spatial_dim
    
    Example: B=32, L=50, D=2 → 12.8 KB per batch (negligible)
    But activations during backprop: B × L × D × hidden_dim × layers
    Example: B=32, L=50, D=2, hidden=128, depth=3 → ~96 MB

Typical training:
    - 1000 samples, sequence_length=20, batch=32 → 30 steps/epoch
    - 100 epochs → 3000 gradient updates
    - Time: 1-10 minutes on modern GPU depending on model size

===================================================================================
ERROR HANDLING
===================================================================================

ValueError:
    - Empty dataset → check trajectory_bundles is non-empty
    - Invalid sequence_length → must be > 0 and < trajectory length

RuntimeError:
    - CUDA out of memory → reduce batch_size or sequence_length
    - NaN loss → check input normalization, reduce learning rate

===================================================================================
"""

# Original docstring
"""Training datasets and utilities for the Flow modeling stack."""
