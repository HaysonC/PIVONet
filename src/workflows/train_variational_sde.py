"""Train the variational SDE controller on CFD trajectories using a pretrained CNF."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Sequence

import torch
import questionary
from rich.console import Console
from torch.utils.data import DataLoader, Subset

from src.interfaces.trajectories import TrajectoryResult
from src.modeling.datasets import CFDTrajectorySequenceDataset
from src.modeling.trainers import TrainingArtifacts, train_variational_sde_model
from src.networks.cnf import CNFModel
from src.networks.variational_sde import VariationalSDEModel
from src.utils.paths import project_root
from src.utils.trajectory_io import save_trajectory_bundle
from src.visualization import TrajectoryPlotter, plot_loss_curve, plot_metric_grid
from src.utils.console_gate import prompt_gate


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundles",
        nargs="+",
        default=["data/cfd/trajectories"],
        help="One or more bundle directories/files for training data.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size for trajectory training.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs for the variational SDE model.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for AdamW.")
    parser.add_argument("--n-particles", type=int, default=4, help="Particles per trajectory when approximating the ELBO.")
    parser.add_argument("--n-integration-steps", type=int, default=50, help="Discretization steps for the SDE integrator.")
    parser.add_argument("--obs-std", type=float, default=0.05, help="Observation noise scale.")
    parser.add_argument("--kl-warmup", type=float, default=1.0, help="Weight for the KL term.")
    parser.add_argument("--control-cost-scale", type=float, default=1.0, help="Weight for the control energy penalty.")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader worker processes.")
    parser.add_argument("--ckpt-dir", default="cache/checkpoints/vsde", help="Directory for VSDE checkpoints.")
    parser.add_argument("--artifact-dir", default=None, help="Directory for plots/sample bundles (defaults to <ckpt-dir>/artifacts).")
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Device to train on (auto selects CUDA/MPS when available).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on samples for quick runs.")
    parser.add_argument("--cnf-checkpoint", required=True, help="Path to the pretrained CNF checkpoint.")
    parser.add_argument("--cnf-hidden-dim", type=int, default=128, help="Hidden units used by the CNF backbone.")
    parser.add_argument("--cnf-depth", type=int, default=3, help="Depth used by the CNF backbone.")
    parser.add_argument("--context-dim", type=int, default=3, help="Conditioning dimension for the CNF model.")
    parser.add_argument("--z-dim", type=int, default=2, help="Latent dimensionality for the SDE model.")
    parser.add_argument("--ctx-dim", type=int, default=128, help="Encoder context dimensionality inside the SDE model.")
    parser.add_argument("--diffusion-learnable", action="store_true", help="Learn the diffusion scale instead of fixing it.")
    parser.add_argument("--viz-trajectories", type=int, default=16, help="Number of trajectories to visualize in the generated plots.")
    return parser.parse_args(argv)


def _resolve_paths(specs: Sequence[str]) -> list[Path]:
    root = project_root()
    resolved: list[Path] = []
    for spec in specs:
        path = Path(spec)
        if not path.is_absolute():
            path = (root / path).resolve()
        resolved.append(path)
    return resolved


def _select_device(preference: str) -> str:
    if preference != "auto":
        return preference
    if torch.cuda.is_available():  # pragma: no cover - hardware specific
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # pragma: no cover - hardware specific
        return "mps"
    return "cpu"


def _ensure_path(path_like: str) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = (project_root() / path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _ensure_ckpt_dir(path_like: str) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = (project_root() / path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _artifact_dirs(ckpt_dir: Path, artifact_arg: str | None) -> tuple[Path, Path, Path]:
    base = Path(artifact_arg).expanduser().resolve() if artifact_arg else (ckpt_dir / "artifacts").resolve()
    plots = base / "plots"
    bundles = base / "bundles"
    plots.mkdir(parents=True, exist_ok=True)
    bundles.mkdir(parents=True, exist_ok=True)
    return base, plots, bundles


def _find_existing_checkpoint(ckpt_dir: Path, preferred_name: str) -> Path | None:
    preferred = ckpt_dir / preferred_name
    if preferred.exists():
        return preferred
    candidates = sorted(ckpt_dir.glob("*.pt"))
    if candidates:
        return candidates[-1]
    return None


def _prompt_checkpoint_action(console: Console, ckpt_dir: Path, preferred_name: str, label: str) -> bool:
    checkpoint = _find_existing_checkpoint(ckpt_dir, preferred_name)
    if checkpoint is None:
        return False
    console.print(
        f"[yellow]{label} checkpoint already exists at {checkpoint}. You can reuse it or retrain.[/]"
    )
    if not sys.stdin.isatty():
        console.print("[yellow]Non-interactive session detected; proceeding with retraining.[/]")
        return False
    with prompt_gate():
        selection = questionary.select(
            f"{label} checkpoint found. What would you like to do?",
            choices=[
                questionary.Choice(title="Skip step (reuse cached checkpoint)", value="skip"),
                questionary.Choice(title="Retrain (overwrite checkpoints)", value="retrain"),
            ],
            default="skip",
        ).ask()
    if selection == "skip":
        console.print(f"[green]Skipping {label.lower()} training and reusing {checkpoint.name}.")
        return True
    console.print(f"[cyan]Continuing {label.lower()} training; existing checkpoints may be overwritten.")
    return False


def _collate_variable_length(batch):
    if not batch:
        raise ValueError("Empty batch encountered in VSDE dataloader.")
    max_len = max(sample[0].shape[0] for sample in batch)
    batch_size = len(batch)
    traj_dim = batch[0][0].shape[1]
    trajs = torch.zeros(batch_size, max_len, traj_dim, dtype=batch[0][0].dtype)
    times = torch.zeros(batch_size, max_len, dtype=batch[0][1].dtype)
    masks = torch.zeros(batch_size, max_len, dtype=batch[0][3].dtype)
    contexts = torch.stack([sample[2] for sample in batch], dim=0)

    for idx, (traj, time, _context, mask) in enumerate(batch):
        length = traj.shape[0]
        trajs[idx, :length] = traj
        times[idx, :length] = time
        masks[idx, :length] = mask
        if length < max_len:
            last_time = time[-1] if len(time) else time.new_zeros(())
            times[idx, length:] = last_time

    return trajs, times, contexts, masks


def _load_cnf(checkpoint: Path, cond_dim: int, hidden_dim: int, depth: int) -> CNFModel:
    model = CNFModel(dim=2, cond_dim=cond_dim, hidden_dim=hidden_dim, depth=depth)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    return model


def _progress_factory(console: Console, *, updates_per_epoch: int = 4):
    state: dict[str, int] = {"epoch": 0, "last_step": 0}

    def _cb(epoch: int, total_epochs: int, step: int, total_steps: int, loss: float, stats: dict | None = None) -> None:
        nonlocal state
        if state["epoch"] != epoch:
            console.print(f"[cyan]Epoch {epoch}/{total_epochs}[/]")
            state["epoch"] = epoch
            state["last_step"] = 0
        interval = max(1, total_steps // max(1, updates_per_epoch))
        should_log = step == 1 or step == total_steps or (step - state["last_step"]) >= interval
        if not should_log:
            return
        parts = [f"loss={loss:.4f}"]
        if stats:
            for key, value in stats.items():
                parts.append(f"{key}={value:.4f}")
        console.log(f"[epoch {epoch}/{total_epochs} | step {step}/{total_steps}] {' '.join(parts)}")
        state["last_step"] = step

    return _cb


def _sample_dataset_trajectories(dataset, count: int) -> TrajectoryResult:
    if len(dataset) == 0:
        raise ValueError("Dataset is empty; cannot visualize trajectories.")
    indices = list(range(min(count, len(dataset))))
    trajectories: list[torch.Tensor] = []
    times_ref: torch.Tensor | None = None
    for idx in indices:
        traj, times, _context, _mask = dataset[idx]
        if times_ref is None:
            times_ref = times
        trajectories.append(traj)
    stacked = torch.stack(trajectories, dim=1)
    history = stacked.cpu().numpy()
    times_arr = (times_ref if times_ref is not None else torch.linspace(0.0, 1.0, steps=history.shape[0])).cpu().numpy()
    return TrajectoryResult(history=history, timesteps=times_arr.tolist())


def _render_sample(
    result: TrajectoryResult,
    plots_dir: Path,
    bundles_dir: Path,
    prefix: str,
    console: Console,
) -> None:
    bundle_path = bundles_dir / f"{prefix}.npz"
    saved_bundle = save_trajectory_bundle(result, bundle_path)
    plotter = TrajectoryPlotter(max_particles=min(result.num_particles, 200))
    artifact = plotter.plot(result, output_path=plots_dir / f"{prefix}.png")
    console.print(f"Saved {prefix} bundle to {saved_bundle}")
    console.print(f"Saved {prefix} visualization to {artifact.path}")


def _plot_histories(artifacts: TrainingArtifacts, plots_dir: Path, console: Console) -> None:
    if artifacts.loss_history:
        loss_art = plot_loss_curve(artifacts.loss_history, plots_dir / "vsde_loss_curve.png", title="VSDE training loss")
        console.print(f"Loss curve written to {loss_art.path}")
    else:
        console.print("[yellow]No VSDE loss history captured; skipping loss plot.[/]")
    try:
        metric_art = plot_metric_grid(
            artifacts.metric_history,
            plots_dir / "vsde_metric_grid.png",
            metric_keys=["log_px", "control_cost", "kl_z0"],
            title="VSDE diagnostics",
        )
        console.print(f"Metric grid written to {metric_art.path}")
    except ValueError:
        console.print("[yellow]VSDE metric history lacked the requested keys; skipping metric grid plot.[/]")


def main(argv: Sequence[str] | None = None) -> None:
    console = Console()
    args = _parse_args(argv)
    ckpt_dir = _ensure_ckpt_dir(args.ckpt_dir)
    if _prompt_checkpoint_action(console, ckpt_dir, "vsde_latest.pt", "VSDE"):
        console.print(f"[cyan]Reusing cached VSDE artifacts under {ckpt_dir}.")
        return
    sources = _resolve_paths(args.bundles)
    dataset_full = CFDTrajectorySequenceDataset(sources)
    dataset: CFDTrajectorySequenceDataset | Subset = dataset_full
    if args.limit is not None:
        capped = min(args.limit, len(dataset_full))
        dataset = Subset(dataset_full, list(range(capped)))
        console.print(f"Capped dataset to {capped} trajectory samples.")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=_collate_variable_length,
    )

    ckpt_path = _ensure_path(args.cnf_checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"CNF checkpoint not found: {ckpt_path}")

    console.print(f"Loading CNF checkpoint from {ckpt_path}")
    cnf_model = _load_cnf(ckpt_path, args.context_dim, args.cnf_hidden_dim, args.cnf_depth)
    vsde = VariationalSDEModel(
        cnf=cnf_model,
        z_dim=args.z_dim,
        ctx_dim=args.ctx_dim,
        drift_hidden=args.cnf_hidden_dim,
        diffusion_learnable=args.diffusion_learnable,
    )

    artifact_base, plots_dir, bundles_dir = _artifact_dirs(ckpt_dir, args.artifact_dir)
    device = _select_device(args.device)
    console.print(f"Training variational SDE on device: {device}")
    training_artifacts = train_variational_sde_model(
        vsde,
        dataloader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        ckpt_dir=ckpt_dir,
        n_particles=args.n_particles,
        obs_std=args.obs_std,
        kl_warmup=args.kl_warmup,
        control_cost_scale=args.control_cost_scale,
        n_integration_steps=args.n_integration_steps,
        progress_cb=_progress_factory(console),
    )
    if training_artifacts.last_checkpoint is None:
        console.print("[yellow]No VSDE checkpoint was produced.")
        return
    latest = ckpt_dir / "vsde_latest.pt"
    shutil.copy2(training_artifacts.last_checkpoint, latest)
    console.print(f"Saved latest VSDE checkpoint to {latest}")

    _plot_histories(training_artifacts, plots_dir, console)

    baseline_sample = _sample_dataset_trajectories(dataset_full, args.viz_trajectories)
    _render_sample(baseline_sample, plots_dir, bundles_dir, "vsde_baseline", console)

    viz_loader = DataLoader(
        dataset,
        batch_size=min(args.viz_trajectories, args.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_variable_length,
    )
    viz_batch = next(iter(viz_loader))
    traj_batch, times_batch, context_batch, mask_batch = viz_batch
    traj_batch = traj_batch.to(device=device, dtype=torch.float32)
    times_batch = times_batch.to(device=device, dtype=torch.float32)
    context_batch = context_batch.to(device=device, dtype=torch.float32)
    mask_batch = mask_batch.to(device=device, dtype=torch.float32)
    vsde.eval()
    with torch.no_grad():
        times_out, traj_out, _u = vsde.sample_posterior(
            traj_batch,
            times_batch,
            context_batch,
            mask=mask_batch,
            n_particles=2,
            n_integration_steps=args.n_integration_steps,
        )
    generated = traj_out[:, 0, :, :].cpu().numpy()
    vsde_result = TrajectoryResult(history=generated, timesteps=times_out.cpu().numpy().tolist())
    _render_sample(vsde_result, plots_dir, bundles_dir, "vsde_generated", console)
    console.print(f"Artifacts stored under {artifact_base}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
