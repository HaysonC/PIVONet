"""Neural ODE networks for trajectory modeling and prediction.

===================================================================================
OVERVIEW
===================================================================================
This package implements Physics-Informed neural ODE architectures for learning
dynamics of particle trajectories from CFD simulations. Two main approaches:

1. CNF (Continuous Normalizing Flows):
   Invertible transformations that learn probability densities
   Efficient for generative modeling and likelihood computation
   Uses torchdiffeq for adjoint-method autodiff

2. Variational SDE:
   Probabilistic latent variable models
   Encoder-decoder architecture
   Captures aleatoric and epistemic uncertainty

===================================================================================
SUBMODULE STRUCTURE
===================================================================================

cnf.py:
    ODEFunc - Neural network parameterizing the vector field
    CNFModel - Continuous normalizing flow wrapper
    Features:
      - Trace-based divergence computation (via Hutchinson estimator)
      - Adjoint method for memory efficiency
      - Context conditioning for trajectory embedding

mlp.py:
    MLP - Multilayer perceptron backbone
    Configurable depth, width, activation, dropout
    Used as backbone for both CNF and SDE

encoder.py:
    Encoder - Trajectory sequence encoder
    Reduces trajectory history to fixed-size latent codes
    Pooling strategies: mean, max, attention

variational_sde.py:
    VariationalSDEModel - Probabilistic SDE wrapper
    Reparameterization trick for gradient estimation
    Annealed KL divergence for stable training

===================================================================================
KEY ALGORITHMS
===================================================================================

Continuous Normalizing Flows (CNF):
    - Transforms data via ODE trajectory: dz/dt = f_θ(z, t)
    - Log-probability change: d(log p)/dt = -tr(∂f/∂z)
    - Trace computed via: tr(J) ≈ ε^T · ∇_z (f · ε), ε ~ N(0,I)
    - Total loss = -log p(z_1) averaged over batch
    - Integration via adjoint sensitivity for O(1) memory

Variational SDE:
    - Encoder: trajectory → posterior q(z|x)
    - Decoder: z → predicted trajectory x'
    - Loss = KL(q(z|x) || p(z)) + ||x - x'||²
    - Gradients via reparameterization: z = μ + σ · ε

MLP Backbone:
    - Dense layers with configurable depth/width
    - Batch normalization and dropout for regularization
    - GELU activations (smooth approximation of ReLU)
    - Output scaling parameter for gradient stability

===================================================================================
DATA STRUCTURES
===================================================================================

CNFModel:
    Attributes:
      - dim: spatial dimension (typically 2 for 2D flows)
      - cond_dim: conditioning vector dimension (latent code size)
      - func: ODEFunc instance
    Methods:
      - forward(x, t, context) -> (z_t, log_det_jacobian)
      - _integrate(x, context, tspan) -> (x_t, log_prob_diff)

VariationalSDEModel:
    Attributes:
      - encoder: Trajectory → latent distribution
      - decoder: Latent + time → next position
      - prior_dist: p(z) typically N(0, I)
    Methods:
      - encode(trajectory) -> (μ, σ²)
      - decode(z, t) -> x_t
      - forward(trajectory) -> reconstructed_trajectory

===================================================================================
MEMORY OWNERSHIP
===================================================================================

Model weights: Owned by nn.Module. Moved to device via .to(device).
Gradients: Computed and retained by PyTorch autograd engine. Zeroed via
optimizer.zero_grad() between training steps.

Input tensors: Assumed borrowed from caller. Model does not retain references
after forward pass unless explicitly stored (e.g., for adversarial examples).

Context vectors: Copied internally; modifications by caller do not affect model.

Checkpoints: Serialized via torch.save(model.state_dict(), path). Fully
non-volatile and can be loaded into different device/dtype contexts.

===================================================================================
COMPUTATIONAL COMPLEXITY
===================================================================================

CNFModel forward pass:
    Time: O(dim × hidden_dim × depth × n_integration_steps)
    Space: O(dim × batch_size + param_count)
    With adjoint: Space reduced to O(dim × batch_size)

Training per epoch (batch_size B, traj_length T, dim D):
    Time: O(B × T × D × net_flops) per forward + backward
    Space: O(B × T × D) for activations (adjoint mitigates)

===================================================================================
USAGE EXAMPLES
===================================================================================

# CNF: Forward pass and log-probability
import torch
from src.networks.cnf import CNFModel

model = CNFModel(dim=2, cond_dim=64, hidden_dim=128)
model.eval()

with torch.no_grad():
    z0 = torch.randn(32, 2)  # batch_size=32, dim=2
    context = torch.randn(32, 64)  # trajectory encoding
    tspan = torch.linspace(0, 1, 10)  # integrate 0→1 over 10 steps

    z_t, log_prob_diff = model._integrate(z0, context, tspan)
    # z_t: (32, 2) final position
    # log_prob_diff: (32, 1) change in log probability

# Variational SDE: Reconstruction loss
from src.networks.variational_sde import VariationalSDEModel

model = VariationalSDEModel(trajectory_dim=2, latent_dim=64)
training_trajectories = torch.randn(32, 10, 2)  # (batch, time, dim)

reconstructed, kl_div = model(training_trajectories)
reconstruction_loss = torch.nn.functional.mse_loss(reconstructed, training_trajectories)
loss = reconstruction_loss + 0.01 * kl_div.mean()  # beta-VAE weighting
loss.backward()

===================================================================================
CONSTRAINTS & ASSUMPTIONS
===================================================================================

1. Batch dimension assumed to be first (B, ...)
2. Time dimension in CNF not used for training directly;
   integration spans provided by caller
3. Context vectors should be normalized or scaled appropriately
4. No in-place operations on model parameters during forward pass
5. dtype consistency: all tensors converted to float32 internally
6. Gradient flow requires x.requires_grad=True for adjoint method

===================================================================================
ERROR HANDLING
===================================================================================

Common errors and recovery:

RuntimeError: "Context must be set before integrating the CNF"
    → Call model.func.set_context(context) before _integrate()

ValueError: "tspan must contain at least two time points"
    → Pass tspan with len(tspan) ≥ 2 to _integrate()

NaN/Inf in loss:
    → Check context norm is reasonable (not too large/small)
    → Reduce learning rate or use gradient clipping
    → Verify dt is stable for ODE solver (typically dt < 0.1)

===================================================================================
"""

from __future__ import annotations

from .cnf import CNFModel
from .encoder import DiffusionEncoderNet
from .mlp import MLP

__all__ = ["CNFModel", "DiffusionEncoderNet", "MLP"]
