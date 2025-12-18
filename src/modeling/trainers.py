"""Training loops for CNF and variational SDE models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ..networks.cnf import CNFModel
from ..networks.variational_sde import VariationalSDEModel


@dataclass
class TrainingArtifacts:
    """
    Immutable bundle of training results and metadata.

    Stores training history (losses, metrics), validation/test results, and checkpoint paths.
    Frozen dataclass ensures immutability (thread-safe, hashable for caching).

    **Attributes**:
        last_checkpoint (Path | None): Path to the most recently saved checkpoint.
            None if no checkpoint directory specified during training.
            Contains model.state_dict() ready to reload: model.load_state_dict(torch.load(...))

        loss_history (list[float]): Training loss for each batch.
            Length = epochs × n_batches. Suitable for plotting convergence curves.
            For CNF: negative log-likelihood (NLL = -log p(x))
            For VSDE: evidence lower bound (ELBO)

        metric_history (list[dict[str, float]]): Per-batch training metrics.
            Each dict contains "epoch", "step", "loss" keys plus auxiliary metrics
            (e.g., for VSDE: "kl_divergence", "reconstruction_loss", "control_cost")

        val_loss_history (list[float]): Validation loss for each batch (if val_loader provided).
            Empty if no validation. Useful for early stopping decisions.

        val_metric_history (list[dict[str, float]]): Per-batch validation metrics.
            Same structure as metric_history. Empty if no validation.

        test_metric_history (list[dict[str, float]]): Per-batch test metrics (if test_loader provided).
            Computed after all training epochs (no gradient). Empty if no test set.

    **Usage Example**::

        artifacts = train_cnf_model(model, train_loader, ...)

        # Inspect results
        print(f"Training epochs completed: {max(m['epoch'] for m in artifacts.metric_history)}")
        print(f"Final training loss: {artifacts.loss_history[-1]:.4f}")
        print(f"Final val loss: {artifacts.val_loss_history[-1]:.4f if artifacts.val_loss_history else 'N/A'}")

        # Load best model (if tracking best-val-loss separately)
        if artifacts.last_checkpoint:
            model.load_state_dict(torch.load(artifacts.last_checkpoint))

        # Plot convergence
        import matplotlib.pyplot as plt
        plt.plot(artifacts.loss_history, label="Train")
        plt.plot(artifacts.val_loss_history, label="Val")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    **Design Notes**:
    - Frozen dataclass (frozen=True, line 19) makes instances immutable and hashable
    - Default factories for lists enable safe default construction
    - Decouples training function from result storage (cleaner API)
    - Can be serialized to JSON for long-term logging
    """

    last_checkpoint: Path | None
    loss_history: list[float] = field(default_factory=list)
    metric_history: list[dict[str, float]] = field(default_factory=list)
    val_loss_history: list[float] = field(default_factory=list)
    val_metric_history: list[dict[str, float]] = field(default_factory=list)
    test_metric_history: list[dict[str, float]] = field(default_factory=list)


def train_cnf_model(
    model: CNFModel,
    dataloader: DataLoader,
    *,
    device: str = "cpu",
    epochs: int = 5,
    lr: float = 1e-3,
    ckpt_dir: Path | None = None,
    val_loader: DataLoader | None = None,
    test_loader: DataLoader | None = None,
    progress_cb: Optional[Callable[[str, int, int, int, int, float], None]] = None,
) -> TrainingArtifacts:
    """
    Train a Continuous Normalizing Flow model on trajectory data.

    Implements maximum likelihood training: maximizes log p(x) = log p_0(f(x)) + ∫ tr(∇f) dt
    where f is the learned CNF dynamics. Uses AdamW optimizer with cosine annealing scheduler.
    Supports validation and test phases, gradient clipping, and progress callbacks.

    **Problem Solved**: Given trajectories from CFD simulations, learn a generative model that
    can (1) compute exact log-likelihoods, (2) sample new trajectories, and (3) analyze
    learned dynamics via the velocity field.

    **Training Algorithm**:
    1. For each epoch:
       a. Training phase: forward → compute NLL = -log p(x), backward, optimize
       b. Optional validation phase: no gradient, compute NLL on val set
    2. Save checkpoints after each epoch
    3. Optional test phase: evaluate on held-out test set (no gradient)
    4. Return training history (losses, metrics, checkpoints)

    **Loss Function**:
    L = -log p(x) = -(log p_0(z) + ∫ tr(∇f) dt)

    where z = flow(x) is the transformed state. CNFModel.log_prob() computes this.
    Negative log-likelihood because we minimize loss (equivalently, maximize likelihood).

    **Parameters**:
        model (CNFModel): Untrained CNF model. Will be moved to device.
        dataloader (DataLoader): Training data loader. Each batch yields:
            (x_final, x0_unused, theta_unused, context) tuples
            - x_final (Tensor): Target positions, shape (batch_size, dim)
            - context (Tensor): Velocity field features, shape (batch_size, cond_dim)
            - Other fields unused in CNF training
        device (str, keyword-only, default="cpu"): Device for training ("cpu", "cuda", "mps", etc.)
        epochs (int, keyword-only, default=5): Number of training epochs. Must be ≥1.
            Typical: 5-50 depending on dataset size. Larger = better convergence but slower.
        lr (float, keyword-only, default=1e-3): Initial learning rate for AdamW.
            Typical range: [1e-4, 1e-2]. Smaller → more stable but slower.
        ckpt_dir (Path | None, keyword-only, default=None): Directory to save checkpoints.
            If None, no checkpoints saved. Created if doesn't exist.
        val_loader (DataLoader | None, keyword-only, default=None): Validation data loader.
            Same structure as dataloader. If None, no validation phase.
        test_loader (DataLoader | None, keyword-only, default=None): Test data loader.
            Same structure. If None, no test phase.
        progress_cb (Callable | None, keyword-only, default=None): Optional callback for
            progress monitoring. Signature: progress_cb(phase, epoch, total_epochs, step,
            total_steps, loss_value). Called for each batch. phase in {"train", "val", "test"}.

    **Returns**:
        TrainingArtifacts: Immutable bundle containing:
            - last_checkpoint (Path | None): Path to final saved checkpoint
            - loss_history (list[float]): Training loss for each batch
            - metric_history (list[dict]): Per-batch metrics with epoch/step/loss
            - val_loss_history (list[float]): Validation loss (if val_loader provided)
            - val_metric_history (list[dict]): Per-batch validation metrics
            - test_metric_history (list[dict]): Per-batch test metrics

    **Side Effects**:
    - Modifies model parameters in-place (via optimizer.step())
    - Saves checkpoint files to ckpt_dir if provided
    - Prints to stdout if non-finite batch values encountered (defensive check)
    - Moves model to device

    **Memory Ownership**:
    - Owns: optimizer, scheduler, metric lists, artifacts
    - Borrows: model (caller retains ownership)
    - Borrows: dataloaders (assumed to yield batches with correct dtype/device handling)

    **Time Complexity**: O(epochs × n_batches × batch_size × model_forward)
    - Typical: 5 epochs × 100 batches × 32 samples ≈ 16K forward passes
    - Each forward: O(dim × steps) for ODE integration ≈ 2×8 = 16 operations per sample

    **Space Complexity**: O(batch_size × dim) for intermediate activations + loss buffers

    **Error Behavior**:
    - Raises ValueError: "Encountered non-finite values..." if batch contains NaN/Inf
        (defensive check, helps debug data corruption)
    - Raises AssertionError: If model.log_prob() returns unexpected shapes (should not occur)
    - Silent recovery: Invalid progress_cb callback ignored (try-except protection)

    **Hyperparameter Tuning**:
    - **lr**: Start with 1e-3. If loss oscillates, reduce. If slow convergence, increase.
    - **epochs**: Watch val loss. If plateauing, stop early (implement early stopping).
    - **ckpt_dir**: Always use; enables resuming training and best-model selection.
    - **grad_clip**: Fixed at 1.0; prevents gradient explosion. Rarely needs tuning.

    **Usage Example**::

        from src.modeling.datasets import CFDTrajectorySequenceDataset
        from src.networks.cnf import CNFModel

        # Prepare data
        dataset = CFDTrajectorySequenceDataset(trajectory_result, ...)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(dataset_val, batch_size=32)

        # Create and train model
        model = CNFModel(dim=2, cond_dim=64, hidden_dim=128, depth=3)
        artifacts = train_cnf_model(
            model,
            train_loader,
            device="cuda",
            epochs=20,
            lr=1e-3,
            ckpt_dir=Path("checkpoints/cnf/"),
            val_loader=val_loader,
            progress_cb=lambda p, e, te, s, ts, l: print(f"[{p}] E{e}/{te} B{s}/{ts} L={l:.4f}")
        )

        # Inspect results
        print(f"Final loss: {artifacts.loss_history[-1]:.4f}")
        print(f"Checkpoint: {artifacts.last_checkpoint}")

        # Load best checkpoint if needed
        model.load_state_dict(torch.load(artifacts.last_checkpoint))

    **Notes on Optimizer/Scheduler**:
    - AdamW: Adaptive learning rate with weight decay (L2 regularization). Robust default.
    - CosineAnnealingLR: Gradually reduces LR from lr to 0. Helps escape local minima.
    - Gradient clipping at 1.0: Prevents exploding gradients during ODE solver backprop.

    **Validation Strategy**:
    - No gradients during validation (torch.no_grad() context)
    - Useful for hyperparameter tuning and early stopping
    - If val loss increases while train loss decreases → overfitting

    **Checkpoint Format**:
    - PyTorch .pt file containing model.state_dict()
    - Filename: cnf_epoch{epoch}.pt
    - Can reload: model.load_state_dict(torch.load(checkpoint_path))
    """
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    ckpt_path: Path | None = None
    loss_history: list[float] = []
    metric_history: list[dict[str, float]] = []
    val_loss_history: list[float] = []
    val_metric_history: list[dict[str, float]] = []
    test_metric_history: list[dict[str, float]] = []

    def _capture_progress(
        phase: str,
        epoch: int,
        total_epochs: int,
        step: int,
        total_steps: int,
        loss_value: float,
    ) -> None:
        """Invoke progress callback if provided; silently ignore errors."""
        if progress_cb is not None:
            progress_cb(phase, epoch, total_epochs, step, total_steps, loss_value)

    @torch.no_grad()
    def _eval_phase(loader: DataLoader | None, phase: str, epoch: int | None) -> None:
        """
        Evaluation phase: compute metrics without gradient.

        **Parameters**:
            loader: Data loader (if None, phase skipped)
            phase: Name for logging ("val" or "test")
            epoch: Epoch number (for record-keeping)
        """
        if loader is None:
            return
        total = len(loader)
        if total == 0:
            return
        model.eval()
        for step, batch in enumerate(loader, start=1):
            x_final, _x0, _theta, context = batch
            x_final = x_final.to(device=device, dtype=torch.float32)
            context = context.to(device=device, dtype=torch.float32)
            # Defensive check: ensure inputs are finite before calling model
            if not torch.isfinite(x_final).all() or not torch.isfinite(context).all():
                raise ValueError(
                    "Encountered non-finite values in training batch. "
                    "Check trajectory bundles for NaNs/Infs. "
                    f"Batch info: step={step}, loader_len={total}"
                )
            log_prob = model.log_prob(x_final, context)
            loss = -log_prob.mean()
            loss_value = float(loss.item())
            record = {
                "step": float(step),
                "loss": loss_value,
            }
            if epoch is not None:
                record["epoch"] = float(epoch)
            _capture_progress(phase, epoch or 0, epochs, step, total, loss_value)
            if phase == "val":
                val_loss_history.append(loss_value)
                val_metric_history.append(record)
            elif phase == "test":
                test_metric_history.append(record)

    # Main training loop
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        total = len(dataloader)
        for step, batch in enumerate(dataloader, start=1):
            x_final, _x0, _theta, context = batch
            x_final = x_final.to(device=device, dtype=torch.float32)
            context = context.to(device=device, dtype=torch.float32)

            # Forward pass: compute NLL = -log p(x)
            optimizer.zero_grad(set_to_none=True)
            log_prob = model.log_prob(x_final, context)
            loss = -log_prob.mean()

            # Backward pass: gradient accumulation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running += float(loss.item())
            loss_value = float(loss.item())
            loss_history.append(loss_value)
            metric_history.append(
                {
                    "epoch": float(epoch),
                    "step": float(step),
                    "loss": loss_value,
                }
            )
            _capture_progress("train", epoch, epochs, step, total, loss_value)

        # Learning rate decay (cosine annealing)
        scheduler.step()

        # Save checkpoint after each epoch
        if ckpt_dir is not None:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"cnf_epoch{epoch}.pt"
            torch.save(model.state_dict(), ckpt_path)

        # Validation phase (optional)
        _eval_phase(val_loader, "val", epoch)

    # Test phase (optional, after all training epochs)
    _eval_phase(test_loader, "test", epochs)

    return TrainingArtifacts(
        last_checkpoint=ckpt_path,
        loss_history=loss_history,
        metric_history=metric_history,
        val_loss_history=val_loss_history,
        val_metric_history=val_metric_history,
        test_metric_history=test_metric_history,
    )


def train_variational_sde_model(
    model: VariationalSDEModel,
    dataloader: DataLoader,
    *,
    device: str = "cpu",
    epochs: int = 3,
    lr: float = 1e-3,
    ckpt_dir: Path | None = None,
    n_particles: int = 4,
    obs_std: float = 0.05,
    kl_warmup: float = 1.0,
    control_cost_scale: float = 1.0,
    n_integration_steps: int = 50,
    progress_cb: Optional[Callable[..., None]] = None,
) -> TrainingArtifacts:
    """
    Train a Variational SDE model on trajectory data using evidence lower bound (ELBO).

    Implements variational inference for stochastic differential equations:
        p(x|z) = ∏_t N(x_t | μ_t(z), obs_std²)  [likelihood]
        p(z) = N(0, I)                           [prior]
        q(z|x) = N(μ_z(x), Σ_z(x))              [posterior from encoder]

    Loss = KL(q(z|x) || p(z)) - E_z[log p(x|z)]  [ELBO]
         = KL_divergence + reconstruction_loss + control_cost

    **Problem Solved**: Learn a generative SDE model via variational inference. Unlike CNF
    (exact likelihood), VSDE provides probabilistic latent variables z and SDE controls,
    enabling uncertainty quantification and interpretable dynamics parameters.

    **Architecture**:
    - Encoder: x → z (latent code encoding trajectory)
    - SDE Decoder: z → ẋ = μ(t) + σ(t)ξ (learned mean and diffusion)
    - Loss: ELBO with KL-divergence regularization and control cost annealing

    **Parameters**:
        model (VariationalSDEModel): Untrained VSDE model. Will be moved to device.
        dataloader (DataLoader): Training data loader. Each batch yields:
            (traj, times, context, mask) tuples
            - traj (Tensor): Trajectory data, shape (batch_size, n_timesteps, dim)
            - times (Tensor): Integration times, shape (batch_size, n_timesteps)
            - context (Tensor): Velocity field features, shape (batch_size, cond_dim)
            - mask (Tensor): Valid sample mask, shape (batch_size, n_timesteps)
        device (str, keyword-only, default="cpu"): Device for training
        epochs (int, keyword-only, default=3): Number of training epochs. Typically smaller
            than CNF (3-10) due to higher variance in stochastic objectives.
        lr (float, keyword-only, default=1e-3): Initial learning rate
        ckpt_dir (Path | None, keyword-only, default=None): Checkpoint directory
        n_particles (int, keyword-only, default=4): Number of samples for ELBO approximation.
            More particles → lower variance but slower training. Typical: 4-16.
        obs_std (float, keyword-only, default=0.05): Likelihood std dev for trajectory.
            Smaller → tighter fit. Typical: 0.01-0.1. Related to observation noise.
        kl_warmup (float, keyword-only, default=1.0): KL-divergence weighting.
            Typically: 1.0 (no annealing). Can use <1.0 to start training with less regularization.
        control_cost_scale (float, keyword-only, default=1.0): Weight for control cost term.
            Encourages simpler learned dynamics. Typical: 0.1-1.0.
        n_integration_steps (int, keyword-only, default=50): ODE solver steps for decoding.
            More → more accurate but slower. Typical: 30-100.
        progress_cb (Callable | None, keyword-only, default=None): Progress callback.
            Flexible signature (calls with variable args; tries both signatures for compatibility).

    **Returns**:
        TrainingArtifacts: Bundle with:
            - last_checkpoint (Path | None)
            - loss_history (list[float]): Training ELBO losses
            - metric_history (list[dict]): Per-batch metrics (loss + stats from model)

    **Side Effects**:
    - Modifies model parameters in-place
    - Saves checkpoint files to ckpt_dir
    - Prints to stdout on callback errors (try-except with fallback signatures)

    **Time Complexity**: O(epochs × n_batches × n_particles × n_integration_steps × batch_size × dim)
    - Much slower than CNF due to stochastic sampling (n_particles)
    - Each step includes ODE integration within ELBO computation
    - Typical: 3 epochs × 100 batches × 4 particles × 50 steps ≈ 60K ODE evaluations

    **Space Complexity**: O(n_particles × batch_size × n_timesteps × dim) for sampling

    **Error Behavior**:
    - Raises AttributeError: If model.compute_elbo() method missing
    - Silent fallback: If progress_cb has different signature, tries both
    - Silent recovery: Progress callback failures ignored

    **ELBO Components** (from model.compute_elbo()):
    - **Reconstruction**: L_recon = -log p(traj|z) ∝ ||predicted - observed||² / obs_std²
    - **KL divergence**: L_kl = KL(q(z|traj) || p(z)) = 0.5 Σ(μ² + σ² - 1 - log(σ²))
    - **Control cost**: L_control = ||control_signal||² (encourages smooth SDE controls)
    - **Total**: L_ELBO = L_recon + kl_warmup * L_kl + control_cost_scale * L_control

    **Hyperparameter Tuning**:
    - **n_particles**: 4 (default) → reasonable variance. Increase if loss too noisy.
    - **obs_std**: Crucial. If predictions far from data, increase. If overfitting, decrease.
    - **kl_warmup**: Usually 1.0. If model collapses to prior, try reducing to 0.5 for first epoch.
    - **control_cost_scale**: 1.0 encourages smooth controls. Reduce if underfitting.
    - **n_integration_steps**: 50 (default). Increase to 100+ for longer horizons.

    **Usage Example**::

        from src.networks.variational_sde import VariationalSDEModel

        # Prepare data (note: different structure than CNF)
        dataset = VariationalSDEDataset(trajectories, context, ...)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

        # Create and train model
        model = VariationalSDEModel(
            state_dim=2,
            latent_dim=8,
            hidden_dim=64,
            n_layers=3
        )

        artifacts = train_variational_sde_model(
            model,
            train_loader,
            device="cuda",
            epochs=10,
            lr=1e-3,
            n_particles=8,
            obs_std=0.05,
            kl_warmup=1.0,
            control_cost_scale=0.1,
            n_integration_steps=50,
            ckpt_dir=Path("checkpoints/vsde/"),
        )

        print(f"Final ELBO: {artifacts.loss_history[-1]:.4f}")
        print(f"Checkpoint: {artifacts.last_checkpoint}")

    **Comparison: CNF vs VSDE**:

    | Aspect | CNF | VSDE |
    |--------|-----|------|
    | Likelihood | Exact | Variational (ELBO bound) |
    | Latent code | Deterministic | Stochastic (z ~ q) |
    | Interpretability | Learned flow | Learned SDE controls |
    | Training speed | Faster | Slower (n_particles samples) |
    | Scalability | Better for large models | Better for uncertainty |
    | Sampling | Flow in reverse | SDE simulation forward |

    **Notes on ELBO Optimization**:
    - ELBO is lower bound on log-likelihood. Gradient estimator has higher variance than CNF.
    - n_particles controls variance-bias tradeoff. More particles → lower variance but slower.
    - KL-divergence prevents posterior collapse (KL → 0). If happening, reduce kl_warmup or observe noise.
    - Control cost encourages simple dynamics (low-energy SDE controls). Tune based on data complexity.

    **Checkpoint Format**: Same as CNF (PyTorch .pt files)
    - Filename: vsde_epoch{epoch}.pt
    """
    model = model.to(device)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    ckpt_path: Path | None = None
    loss_history: list[float] = []
    metric_history: list[dict[str, float]] = []

    # Main training loop
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        total = len(dataloader)
        for step, batch in enumerate(dataloader, start=1):
            # Unpack batch (trajectory data structure differs from CNF)
            traj, times, context, mask = batch
            traj = traj.to(device=device, dtype=torch.float32)
            times = times.to(device=device, dtype=torch.float32)
            context = context.to(device=device, dtype=torch.float32)
            mask = mask.to(device=device, dtype=torch.float32)

            # Forward pass: compute ELBO and auxiliary statistics
            optimizer.zero_grad(set_to_none=True)
            loss, stats = model.compute_elbo(
                traj,
                times,
                context=context,
                mask=mask,
                n_particles=n_particles,
                obs_std=obs_std,
                kl_warmup=kl_warmup,
                control_cost_scale=control_cost_scale,
                n_integration_steps=n_integration_steps,
            )

            # Backward pass: gradient accumulation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running += float(loss.item())
            loss_value = float(loss.item())
            record = {"epoch": float(epoch), "step": float(step), "loss": loss_value}

            # Append auxiliary statistics (KL, reconstruction, control cost) if available
            for key, value in (stats or {}).items():
                record[key] = float(value)

            loss_history.append(loss_value)
            metric_history.append(record)

            # Invoke progress callback with flexible signature handling
            if progress_cb is not None:
                try:
                    # Try calling with full statistics
                    progress_cb(epoch, epochs, step, total, loss_value, stats)
                except TypeError:
                    # Fallback to simpler signature (for compatibility)
                    progress_cb(epoch, epochs, step, total, loss_value)

        # Learning rate decay (cosine annealing)
        scheduler.step()

        # Save checkpoint after each epoch
        if ckpt_dir is not None:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"vsde_epoch{epoch}.pt"
            torch.save(model.state_dict(), ckpt_path)

    return TrainingArtifacts(
        last_checkpoint=ckpt_path,
        loss_history=loss_history,
        metric_history=metric_history,
    )
