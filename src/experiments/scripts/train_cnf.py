"""Train the CNF model on CFD trajectory bundles for orchestrated experiments."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Sequence

import torch
import questionary
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader, Subset, random_split

from src.interfaces.trajectories import TrajectoryResult
from src.modeling.datasets import CFDEndpointDataset, CFDTrajectorySequenceDataset
from src.modeling.trainers import TrainingArtifacts, train_cnf_model
from src.networks.cnf import CNFModel
from src.utils.paths import project_root
from src.utils.trajectory_io import save_trajectory_bundle
from src.visualization import TrajectoryPlotter, plot_loss_curve


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundles",
        nargs="+",
        default=["data/cfd/trajectories"],
        help="One or more bundle directories/files to use for training.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for the endpoint dataset.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs for the CNF model.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for AdamW."
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=128, help="Hidden units inside the CNF MLP."
    )
    parser.add_argument(
        "--depth", type=int, default=3, help="Number of residual blocks in the CNF MLP."
    )
    parser.add_argument(
        "--context-dim",
        type=int,
        default=3,
        help="Context dimensionality emitted by the dataset.",
    )
    parser.add_argument(
        "--workers", type=int, default=0, help="DataLoader worker processes."
    )
    parser.add_argument(
        "--ckpt-dir",
        default="cache/checkpoints/cnf",
        help="Directory to store CNF checkpoints.",
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="Directory for plots and sample bundles (defaults to <ckpt-dir>/artifacts).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Device to train on (auto selects CUDA/MPS when available).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of samples for fast runs.",
    )
    parser.add_argument(
        "--viz-trajectories",
        type=int,
        default=32,
        help="Number of trajectories to include in the visualization bundle.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of samples reserved for validation.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of samples reserved for held-out testing.",
    )
    parser.add_argument(
        "--split-seed", type=int, default=42, help="Random seed for dataset splitting."
    )
    parser.add_argument(
        "--progress-mode",
        choices=("auto", "bars", "plain"),
        default="auto",
        help="Display style for training progress (bars require an interactive terminal).",
    )
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
    if (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):  # pragma: no cover - hardware specific
        return "mps"
    return "cpu"


def _ensure_ckpt_dir(path_like: str) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = (project_root() / path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _artifact_dirs(ckpt_dir: Path, artifact_dir: str | None) -> tuple[Path, Path, Path]:
    base = (
        Path(artifact_dir).expanduser().resolve()
        if artifact_dir
        else (ckpt_dir / "artifacts").resolve()
    )
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


def _prompt_checkpoint_action(
    console: Console, ckpt_dir: Path, preferred_name: str, label: str
) -> bool:
    checkpoint = _find_existing_checkpoint(ckpt_dir, preferred_name)
    if checkpoint is None:
        return False
    console.print(
        f"[yellow]{label} checkpoint already exists at {checkpoint}. You can reuse it or retrain.[/]"
    )
    if not sys.stdin.isatty():
        console.print(
            "[yellow]Non-interactive session detected; proceeding with retraining.[/]"
        )
        return False
    selection = questionary.select(
        f"{label} checkpoint found. What would you like to do?",
        choices=[
            questionary.Choice(
                title="Skip step (reuse cached checkpoint)", value="skip"
            ),
            questionary.Choice(
                title="Retrain (overwrite checkpoints)", value="retrain"
            ),
        ],
        default="skip",
    ).ask()
    if selection == "skip":
        console.print(
            f"[green]Skipping {label.lower()} training and reusing {checkpoint.name}."
        )
        return True
    console.print(
        f"[cyan]Continuing {label.lower()} training; existing checkpoints may be overwritten."
    )
    return False


class TrainingProgressDisplay:
    """Render dual progress bars for epochs and intra-epoch batches."""

    def __init__(self, console: Console, total_epochs: int) -> None:
        self.console = console
        self.total_epochs = total_epochs
        self.progress = Progress(
            TextColumn("{task.description}", justify="right"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        )
        self.epoch_task: TaskID | None = None
        self.batch_task: TaskID | None = None
        self.current_phase: str | None = None
        self.current_total: int = 1

    def __enter__(self) -> "TrainingProgressDisplay":
        self.progress.__enter__()
        self.epoch_task = self.progress.add_task(
            "[cyan]overall training", total=self.total_epochs
        )
        self.batch_task = self.progress.add_task("[magenta]epoch batches", total=1)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.progress.__exit__(exc_type, exc, tb)

    def callback(
        self,
        phase: str,
        epoch: int,
        total_epochs: int,
        step: int,
        total_steps: int,
        loss: float,
    ) -> None:
        if self.batch_task is None or self.epoch_task is None:
            return
        total_steps = max(1, total_steps)
        if self.current_phase != phase or self.current_total != total_steps:
            self.current_phase = phase
            self.current_total = total_steps
            self.progress.reset(self.batch_task, total=total_steps)
        description = f"[{phase}] epoch {epoch}/{total_epochs} | batch {step}/{total_steps} | loss {loss:.4f}"
        self.progress.update(
            self.batch_task,
            completed=min(step, total_steps),
            description=description,
        )
        if phase == "train" and step >= total_steps:
            self.progress.update(
                self.epoch_task, completed=min(epoch, self.total_epochs)
            )


class PlainProgressLogger:
    """Fallback progress reporter when fancy bars are disabled."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self._start_time: float | None = None
        self._train_total: int | None = None
        self._train_completed: int = 0

    def __enter__(self) -> "PlainProgressLogger":
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401 - no cleanup
        return None

    def callback(
        self,
        phase: str,
        epoch: int,
        total_epochs: int,
        step: int,
        total_steps: int,
        loss: float,
    ) -> None:
        total_steps = max(1, total_steps)
        eta_text = self._eta_text(phase, epoch, total_epochs, step, total_steps)
        self.console.log(
            f"[{phase}] epoch {epoch}/{total_epochs} | batch {step}/{total_steps} | loss={loss:.4f}{eta_text}"
        )

    def _eta_text(
        self, phase: str, epoch: int, total_epochs: int, step: int, total_steps: int
    ) -> str:
        if phase == "train":
            self._track_train_progress(epoch, total_epochs, step, total_steps)
        remaining = self._estimate_remaining_seconds()
        if remaining is None:
            return ""
        return f" | ETA {self._format_duration(remaining)}"

    def _track_train_progress(
        self, epoch: int, total_epochs: int, step: int, total_steps: int
    ) -> None:
        epoch = max(epoch, 1)
        total_epochs = max(total_epochs, 1)
        completed_within_epoch = min(step, total_steps)
        total_seen = (epoch - 1) * total_steps + completed_within_epoch
        grand_total = total_epochs * total_steps
        self._train_total = grand_total
        self._train_completed = max(self._train_completed, min(total_seen, grand_total))

    def _estimate_remaining_seconds(self) -> float | None:
        if not self._start_time or not self._train_total or self._train_completed <= 0:
            return None
        remaining_units = max(self._train_total - self._train_completed, 0)
        if remaining_units == 0:
            return 0.0
        elapsed = max(time.time() - self._start_time, 0.0)
        completed = max(self._train_completed, 1)
        avg_unit_duration = elapsed / completed if elapsed > 0 else None
        if avg_unit_duration is None or avg_unit_duration <= 0:
            return None
        return remaining_units * avg_unit_duration

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0.0, seconds)
        hours, remainder = divmod(int(seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours:d}h {minutes:02d}m"
        if minutes:
            return f"{minutes:d}m {secs:02d}s"
        return f"{secs:d}s"


def _split_endpoint_dataset(
    dataset: CFDEndpointDataset | Subset,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    console: Console,
) -> tuple[Subset, Subset, Subset]:
    if val_ratio < 0 or test_ratio < 0:
        raise ValueError("Validation/test ratios must be non-negative.")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError(
            "Validation + test ratio must sum to less than 1.0 to retain training data."
        )
    total = len(dataset)
    if total < 3:
        raise ValueError("Need at least three samples to create train/val/test splits.")
    val_len = int(total * val_ratio)
    test_len = int(total * test_ratio)
    if val_ratio > 0 and val_len == 0:
        val_len = 1
    if test_ratio > 0 and test_len == 0:
        test_len = 1
    train_len = total - val_len - test_len
    if train_len <= 0:
        raise ValueError(
            "Split ratios leave no training samples; decrease validation/test ratios."
        )
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_len, val_len, test_len], generator=generator
    )
    console.print(
        f"Split dataset -> train: {train_len} | val: {val_len} | test: {test_len}"
    )
    return train_ds, val_ds, test_ds


def _build_dataloaders(
    train_ds: Subset,
    val_ds: Subset,
    test_ds: Subset,
    batch_size: int,
    workers: int,
) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=workers
    )
    val_loader = (
        None
        if len(val_ds) == 0
        else DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=workers
        )
    )
    test_loader = (
        None
        if len(test_ds) == 0
        else DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=workers
        )
    )
    return train_loader, val_loader, test_loader


def _loss_summary(losses: Sequence[float]) -> dict[str, float]:
    if not losses:
        return {}
    count = len(losses)
    avg = float(sum(losses) / count)
    return {
        "count": float(count),
        "min": float(min(losses)),
        "max": float(max(losses)),
        "final": float(losses[-1]),
        "mean": avg,
    }


def _dump_metrics_json(
    artifacts: TrainingArtifacts,
    output_path: Path,
    *,
    dataset_sizes: dict[str, int],
    hyperparams: dict[str, object],
    device: str,
    started_at: float,
    finished_at: float,
) -> Path:
    record = {
        "device": device,
        "dataset_sizes": dataset_sizes,
        "hyperparameters": hyperparams,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_sec": finished_at - started_at,
        "train_loss_history": artifacts.loss_history,
        "val_loss_history": artifacts.val_loss_history,
        "train_metrics": [
            {"phase": "train", **entry} for entry in artifacts.metric_history
        ],
        "val_metrics": [
            {"phase": "val", **entry} for entry in artifacts.val_metric_history
        ],
        "test_metrics": [
            {"phase": "test", **entry} for entry in artifacts.test_metric_history
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(record, fp, indent=2)
    return output_path


def _sample_training_trajectories(
    dataset: CFDTrajectorySequenceDataset,
    count: int,
) -> TrajectoryResult:
    if len(dataset) == 0:
        raise ValueError(
            "Trajectory sequence dataset is empty; cannot create visualization sample."
        )
    indices = list(range(min(count, len(dataset))))
    trajectories: list[torch.Tensor] = []
    times_ref: torch.Tensor | None = None
    for idx in indices:
        traj, times, _context, _mask = dataset[idx]
        if times_ref is None:
            times_ref = times
        trajectories.append(traj)
    stacked = torch.stack(trajectories, dim=1)  # (T, N, 2)
    history = stacked.cpu().numpy()
    times_arr = (
        (
            times_ref
            if times_ref is not None
            else torch.linspace(0.0, 1.0, steps=history.shape[0])
        )
        .cpu()
        .numpy()
    )
    return TrajectoryResult(history=history, timesteps=times_arr.tolist())


def _render_trajectory_sample(
    result: TrajectoryResult, plots_dir: Path, bundles_dir: Path, console: Console
) -> None:
    bundle_path = bundles_dir / "cnf_training_sample.npz"
    saved_bundle = save_trajectory_bundle(result, bundle_path)
    plotter = TrajectoryPlotter(max_particles=min(result.num_particles, 200))
    artifact = plotter.plot(result, output_path=plots_dir / "cnf_training_sample.png")
    console.print(f"Saved sample trajectory bundle to {saved_bundle}")
    console.print(f"Saved trajectory visualization to {artifact.path}")


def _plot_losses(
    artifacts: TrainingArtifacts, plots_dir: Path, console: Console
) -> None:
    if artifacts.loss_history:
        artifact = plot_loss_curve(
            artifacts.loss_history,
            plots_dir / "cnf_train_loss.png",
            title="CNF training loss",
        )
        console.print(f"Training loss curve written to {artifact.path}")
    else:
        console.print(
            "[yellow]No training loss history recorded; skipping training loss plot.[/]"
        )
    if artifacts.val_loss_history:
        val_artifact = plot_loss_curve(
            artifacts.val_loss_history,
            plots_dir / "cnf_val_loss.png",
            title="CNF validation loss",
        )
        console.print(f"Validation loss curve written to {val_artifact.path}")
    else:
        console.print(
            "[yellow]No validation loss history recorded; skipping validation loss plot.[/]"
        )


def main(argv: Sequence[str] | None = None) -> None:
    console = Console()
    args = _parse_args(argv)
    ckpt_dir = _ensure_ckpt_dir(args.ckpt_dir)
    if _prompt_checkpoint_action(console, ckpt_dir, "cnf_latest.pt", "CNF"):
        console.print(f"[cyan]Reusing cached CNF artifacts under {ckpt_dir}.")
        return
    sources = _resolve_paths(args.bundles)
    console.print(f"Loading bundles from: {[str(s) for s in sources]}")
    dataset_full = CFDEndpointDataset(sources)
    dataset: CFDEndpointDataset | Subset = dataset_full
    if args.limit is not None:
        capped = min(args.limit, len(dataset_full))
        dataset = Subset(dataset_full, list(range(capped)))
        console.print(f"Capped dataset to {capped} samples for quick experimentation.")
    train_ds, val_ds, test_ds = _split_endpoint_dataset(
        dataset, args.val_ratio, args.test_ratio, args.split_seed, console
    )
    train_loader, val_loader, test_loader = _build_dataloaders(
        train_ds, val_ds, test_ds, args.batch_size, args.workers
    )
    dataset_sizes = {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)}

    model = CNFModel(
        dim=2, cond_dim=args.context_dim, hidden_dim=args.hidden_dim, depth=args.depth
    )
    artifact_base, plots_dir, bundles_dir = _artifact_dirs(ckpt_dir, args.artifact_dir)
    device = _select_device(args.device)
    console.print(f"Training CNF on device: {device}")
    run_start = time.time()
    env_progress = os.getenv("FLOW_PROGRESS_MODE")
    resolved_progress = args.progress_mode
    if resolved_progress == "auto" and env_progress in {"bars", "plain"}:
        resolved_progress = env_progress
    if resolved_progress == "auto":
        use_bars = console.is_terminal and sys.stdout.isatty()
    else:
        use_bars = resolved_progress == "bars"
    progress_context = (
        TrainingProgressDisplay(console, args.epochs)
        if use_bars
        else PlainProgressLogger(console)
    )
    with progress_context as progress_display:
        cb = getattr(progress_display, "callback", None)
        training_artifacts = train_cnf_model(
            model,
            train_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            ckpt_dir=ckpt_dir,
            val_loader=val_loader,
            test_loader=test_loader,
            progress_cb=cb,
        )
    run_end = time.time()
    if training_artifacts.last_checkpoint is None:
        console.print("[yellow]Training finished but no checkpoint was returned.[/]")
        return
    latest = ckpt_dir / "cnf_latest.pt"
    shutil.copy2(training_artifacts.last_checkpoint, latest)
    console.print(f"Saved latest CNF checkpoint to {latest}")

    train_summary = _loss_summary(training_artifacts.loss_history)
    val_summary = _loss_summary(training_artifacts.val_loss_history)
    if train_summary:
        console.print(
            f"Training loss -> final {train_summary['final']:.4f} | best {train_summary['min']:.4f} over {int(train_summary['count'])} steps"
        )
    if val_summary:
        console.print(
            f"Validation loss -> final {val_summary['final']:.4f} | best {val_summary['min']:.4f} over {int(val_summary['count'])} steps"
        )
    if training_artifacts.test_metric_history:
        test_losses = [
            entry["loss"] for entry in training_artifacts.test_metric_history
        ]
        test_avg = float(sum(test_losses) / len(test_losses))
        console.print(
            f"Test split average loss: {test_avg:.4f} ({len(test_losses)} batches)"
        )

    _plot_losses(training_artifacts, plots_dir, console)

    hyperparams = {key: getattr(args, key) for key in vars(args)}
    metrics_path = artifact_base / "cnf_training_metrics.json"
    metrics_file = _dump_metrics_json(
        training_artifacts,
        metrics_path,
        dataset_sizes=dataset_sizes,
        hyperparams=hyperparams,
        device=device,
        started_at=run_start,
        finished_at=run_end,
    )
    console.print(f"Logged training metrics to {metrics_file}")

    seq_dataset = CFDTrajectorySequenceDataset(sources)
    sample = _sample_training_trajectories(seq_dataset, args.viz_trajectories)
    _render_trajectory_sample(sample, plots_dir, bundles_dir, console)
    console.print(f"Artifacts stored under {artifact_base}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
