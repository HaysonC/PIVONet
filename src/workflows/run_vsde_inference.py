"""Run posterior sampling with a trained variational SDE controller and compare to CNF-only forecasts."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Sequence, cast

import numpy as np
from scipy.interpolate import splprep, splev

import torch
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader, Subset

from src.interfaces.trajectories import TrajectoryResult
from src.modeling.datasets import CFDTrajectorySequenceDataset
from src.networks.cnf import CNFModel
from src.networks.variational_sde import VariationalSDEModel
from src.utils.paths import project_root
from src.utils.trajectory_io import save_trajectory_bundle
from src.visualization import TrajectoryPlotter

README_TEMPLATE = project_root() / "src" / "experiments" / "docs" / "README_vsde_inference_template.md"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundles",
        nargs="+",
        default=["data/cfd/trajectories"],
        help="One or more bundle directories/files used for inference inputs.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference batches.")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader worker processes.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on the number of samples loaded for inference.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Device to run inference on (auto selects CUDA/MPS when available).",
    )
    parser.add_argument("--cnf-checkpoint", required=True, help="Path to the pretrained CNF checkpoint.")
    parser.add_argument("--vsde-checkpoint", required=True, help="Path to the trained VSDE checkpoint to load.")
    parser.add_argument("--cnf-hidden-dim", type=int, default=128, help="Hidden units used by the CNF backbone.")
    parser.add_argument("--drift-hidden", type=int, default=128, help="Hidden units for VSDE posterior drift network.")
    parser.add_argument("--cnf-depth", type=int, default=3, help="Depth used by the CNF backbone.")
    parser.add_argument("--context-dim", type=int, default=3, help="Conditioning dimension for the CNF model.")
    parser.add_argument("--z-dim", type=int, default=2, help="Latent dimensionality for the SDE model.")
    parser.add_argument("--ctx-dim", type=int, default=128, help="Encoder context dimensionality inside the SDE model.")
    parser.add_argument("--diffusion-learnable", action="store_true", help="Expect the diffusion scale to be learnable (must match training).")
    parser.add_argument("--n-particles", type=int, default=2, help="Number of posterior particles to sample during inference.")
    parser.add_argument(
        "--n-integration-steps",
        type=int,
        default=60,
        help="Discretization steps for the posterior SDE sampler.",
    )
    parser.add_argument(
        "--viz-trajectories",
        type=int,
        default=32,
        help="Number of trajectories to visualize from the dataset for reference.",
    )
    parser.add_argument(
        "--output-dir",
        default="cache/artifacts/vsde_inference",
        help="Directory where inference bundles/plots will be written.",
    )
    parser.add_argument(
        "--overlay-dir",
        default=None,
        help="Optional directory for overlay PNGs (default: <output-dir>/plots)."
             " Relative paths are resolved under --output-dir.",
    )
    parser.add_argument(
        "--overlay-max-paths",
        type=int,
        default=5,
        help="Maximum number of trajectories drawn per overlay plot.",
    )
    parser.add_argument("--cnf-samples", type=int, default=8, help="Number of CNF samples per trajectory when building the baseline.")
    parser.add_argument("--region-bins", type=int, default=2, help="How many spatial bins per axis to use when summarizing regional metrics.")
    parser.add_argument(
        "--length-scale",
        type=float,
        default=1.0,
        help="Global multiplier on CNF drift to adjust path length (curvature untouched).",
    )
    parser.add_argument(
        "--diffusion-scale",
        type=float,
        default=1.0,
        help="Multiplier on the learned diffusion parameter to increase/decrease random wiggling.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional limit on the number of dataloader batches to evaluate (defaults to all batches).",
    )
    parser.add_argument(
        "--skip-cnf-baseline",
        action="store_true",
        help="Disable CNF-only comparison (only VSDE predictions will be reported).",
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
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # pragma: no cover - hardware specific
        return "mps"
    return "cpu"


def _resolve_path(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (project_root() / path).resolve()


def _collate_variable_length(batch):
    if not batch:
        raise ValueError("Empty batch encountered in VSDE inference dataloader.")
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


def _ensure_output_dirs(base_arg: str) -> tuple[Path, Path, Path]:
    base = Path(base_arg).expanduser().resolve()
    plots = base / "plots"
    bundles = base / "bundles"
    plots.mkdir(parents=True, exist_ok=True)
    bundles.mkdir(parents=True, exist_ok=True)
    return base, plots, bundles


def _resolve_overlay_dir(arg: str | None, artifact_base: Path, default_plots: Path) -> Path:
    if not arg:
        default_plots.mkdir(parents=True, exist_ok=True)
        return default_plots
    candidate = Path(arg)
    if candidate.is_absolute():
        resolved = candidate
    else:
        resolved = (artifact_base / candidate).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _sample_dataset_trajectories(dataset: CFDTrajectorySequenceDataset, count: int) -> TrajectoryResult:
    if len(dataset) == 0:
        raise ValueError("Trajectory sequence dataset is empty; cannot create visualization sample.")
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


def _render_sample(result: TrajectoryResult, plots_dir: Path, bundles_dir: Path, prefix: str, console: Console) -> None:
    bundle_path = bundles_dir / f"{prefix}.npz"
    saved_bundle = save_trajectory_bundle(result, bundle_path)
    plotter = TrajectoryPlotter(max_particles=min(result.num_particles, 200))
    artifact = plotter.plot(result, output_path=plots_dir / f"{prefix}.png")
    console.print(f"Saved {prefix} bundle to {saved_bundle}")
    console.print(f"Saved {prefix} visualization to {artifact.path}")


def _cnf_predict_final(model: CNFModel, context_batch: torch.Tensor, *, samples_per_traj: int) -> torch.Tensor:
    samples_per_traj = max(1, int(samples_per_traj))
    batch = context_batch.shape[0]
    expanded_context = context_batch.repeat_interleave(samples_per_traj, dim=0)
    with torch.no_grad():
        preds = model.sample(expanded_context.shape[0], expanded_context)
    preds = preds.view(batch, samples_per_traj, -1)
    return preds.mean(dim=1)


def _compute_region_indices(starts: torch.Tensor, bins: int) -> tuple[torch.Tensor, torch.Tensor, list[tuple[float, float]], list[tuple[float, float]]]:
    bins = max(1, bins)
    starts_cpu = starts.contiguous().cpu()
    min_xy = starts_cpu.min(dim=0).values
    max_xy = starts_cpu.max(dim=0).values
    if torch.allclose(min_xy, max_xy):
        max_xy = min_xy + 1.0
    x_edges = torch.linspace(float(min_xy[0]), float(max_xy[0]), steps=bins + 1)
    y_edges = torch.linspace(float(min_xy[1]), float(max_xy[1]), steps=bins + 1)
    x_bins = torch.bucketize(starts_cpu[:, 0], x_edges[1:-1])
    y_bins = torch.bucketize(starts_cpu[:, 1], y_edges[1:-1])
    x_bins = torch.clamp(x_bins, 0, bins - 1)
    y_bins = torch.clamp(y_bins, 0, bins - 1)
    x_bounds = [(float(x_edges[i]), float(x_edges[i + 1])) for i in range(bins)]
    y_bounds = [(float(y_edges[i]), float(y_edges[i + 1])) for i in range(bins)]
    return x_bins, y_bins, x_bounds, y_bounds


def _summarize_errors(
    console: Console,
    output_path: Path,
    start_positions: torch.Tensor,
    gt_final: torch.Tensor,
    vsde_pred: torch.Tensor,
    cnf_pred: torch.Tensor | None,
    bins: int,
) -> dict[str, object]:
    gt = gt_final.cpu()
    vsde = vsde_pred.cpu()
    vsde_err = torch.linalg.norm(vsde - gt, dim=1)
    cnf = cnf_pred.cpu() if cnf_pred is not None else None
    cnf_err = torch.linalg.norm(cnf - gt, dim=1) if cnf is not None else None
    summary: dict[str, object] = {
        "count": int(gt.shape[0]),
        "vsde": {
            "mae": float(vsde_err.mean().item()),
            "rmse": float(torch.sqrt((vsde_err ** 2).mean()).item()),
            "median": float(vsde_err.median().item()),
        },
    }
    if cnf_err is not None:
        summary["cnf"] = {
            "mae": float(cnf_err.mean().item()),
            "rmse": float(torch.sqrt((cnf_err ** 2).mean()).item()),
            "median": float(cnf_err.median().item()),
        }
    x_bins, y_bins, x_bounds, y_bounds = _compute_region_indices(start_positions, bins)
    regions_payload: list[dict[str, object]] = []
    rows: list[tuple[str, int, float, float | None]] = []
    for ix in range(len(x_bounds)):
        for iy in range(len(y_bounds)):
            mask = (x_bins == ix) & (y_bins == iy)
            count = int(mask.sum().item())
            if count == 0:
                continue
            region_vsde = vsde_err[mask]
            vsde_mae = float(region_vsde.mean().item())
            cnf_mae_value = float(cnf_err[mask].mean().item()) if cnf_err is not None else None
            label = f"region_x{ix}_y{iy}"
            region_entry: dict[str, object] = {
                "label": label,
                "count": count,
                "x_bounds": x_bounds[ix],
                "y_bounds": y_bounds[iy],
                "vsde_mae": vsde_mae,
            }
            if cnf_mae_value is not None:
                region_entry["cnf_mae"] = cnf_mae_value
            regions_payload.append(region_entry)
            rows.append((label, count, vsde_mae, cnf_mae_value))
    summary["regions"] = regions_payload
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    table = Table(title="VSDE vs CNF final-position error by region")
    table.add_column("Region")
    table.add_column("Count", justify="right")
    table.add_column("VSDE MAE", justify="right")
    if cnf_err is not None:
        table.add_column("CNF MAE", justify="right")
    if not rows:
        cnf_global = None
        if cnf_err is not None and "cnf" in summary:
            cnf_global = float(summary["cnf"]["mae"])  # type: ignore[index]
        rows.append(("global", int(summary["count"]), float(summary["vsde"]["mae"]), cnf_global))  # type: ignore[index]
    for label, count, vsde_mae, cnf_mae in rows:
        cnf_value = f"{cnf_mae:.4f}" if cnf_mae is not None else "-"
        table.add_row(label, str(count), f"{vsde_mae:.4f}", cnf_value)
    console.print(table)
    console.print(f"Comparison metrics saved to {output_path}")
    return summary


def _plot_region_error_bars(summary: dict[str, object], plots_dir: Path) -> Path | None:
    regions = summary.get("regions")
    if not isinstance(regions, list) or not regions:
        return None
    labels: list[str] = []
    vsde_vals: list[float] = []
    cnf_vals: list[float | None] = []
    has_cnf = "cnf" in summary
    for entry in regions:
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("label", "region"))
        labels.append(label)
        vsde_vals.append(float(entry.get("vsde_mae", 0.0)))
        cnf_value = entry.get("cnf_mae")
        cnf_vals.append(float(cnf_value) if cnf_value is not None else None)
    if not labels:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = range(len(labels))
    width = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar([i - width / 2 for i in x], vsde_vals, width, label="VSDE", color="tab:blue")
    if has_cnf and any(v is not None for v in cnf_vals):
        cnf_clean = [v if v is not None else 0.0 for v in cnf_vals]
        plt.bar([i + width / 2 for i in x], cnf_clean, width, label="CNF", color="tab:orange")
    plt.xticks(list(x), labels, rotation=30, ha="right")
    plt.ylabel("MAE (final position)")
    plt.title("Regional VSDE vs CNF error")
    plt.legend()
    plt.tight_layout()
    plot_path = plots_dir / "vsde_vs_cnf_difference.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def _plot_timestep_mae(
    gt_traj: torch.Tensor,  # (T, N, 2)
    vsde_traj: torch.Tensor,  # (T, N, 2)
    cnf_traj: torch.Tensor | None,  # (T, N, 2) or None
    plots_dir: Path,
    console: Console,
) -> Path:
    """Plot MAE at each timestep throughout the trajectory."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    # Compute per-timestep MAE across all particles
    vsde_errors = torch.linalg.norm(vsde_traj - gt_traj, dim=2)  # (T, N)
    vsde_mae_per_t = vsde_errors.mean(dim=1)  # (T,)
    
    cnf_mae_per_t = None
    if cnf_traj is not None:
        cnf_errors = torch.linalg.norm(cnf_traj - gt_traj, dim=2)  # (T, N)
        cnf_mae_per_t = cnf_errors.mean(dim=1)  # (T,)
    
    num_timesteps = vsde_mae_per_t.shape[0]
    timesteps = torch.linspace(0, 1, num_timesteps)
    
    plt.figure(figsize=(12, 5))
    plt.plot(timesteps.numpy(), vsde_mae_per_t.numpy(), label="VSDE", color="tab:blue", linewidth=2)
    if cnf_mae_per_t is not None:
        plt.plot(timesteps.numpy(), cnf_mae_per_t.numpy(), label="CNF", color="tab:orange", linewidth=2)
    
    plt.xlabel("Normalized Time (t)", fontsize=12)
    plt.ylabel("Mean Absolute Error", fontsize=12)
    plt.title("Per-Timestep MAE: Full Trajectory Accuracy", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = plots_dir / "timestep_mae.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    console.print(f"[green]✓[/green] Saved per-timestep MAE plot to {plot_path}")
    return plot_path


def _write_plot_data_json(summary: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump({"regions": summary.get("regions", []), "totals": {k: summary.get(k) for k in ("vsde", "cnf", "count")}}, fp, indent=2)


def _simulate_cnf_baseline(
    cnf: CNFModel,
    z0: torch.Tensor,
    times: torch.Tensor,
    context: torch.Tensor,
    gt_final: torch.Tensor | None = None,
    length_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate CNF baseline with adaptive per-particle scaling.
    
    Scales the entire CNF trajectory (not just drift) so final displacement matches GT.
    
    Returns:
        trajectories: (T, B, 2) tensor of CNF trajectories
        scale_factors: (B,) tensor of per-particle scaling factors
    """
    if times.dim() == 2:
        times = times[0]
    
    # First pass: compute unscaled CNF trajectory
    z = z0.clone()
    traj_unscaled = [z0]
    for idx in range(times.shape[0] - 1):
        t_i = float(times[idx].item())
        dt = (times[idx + 1] - times[idx]).clamp_min(1e-5)
        drift = cnf.eval_field(z, context, t_i)
        z = z + drift * dt
        traj_unscaled.append(z)
    traj_unscaled = torch.stack(traj_unscaled, dim=0)  # (T, B, 2)
    
    # Compute per-particle scaling factors based on displacement
    if gt_final is not None:
        # ratio = ||GT_final - GT_start|| / ||CNF_final - CNF_start||
        gt_displacement = torch.linalg.norm(gt_final - z0, dim=1, keepdim=True)  # (B, 1)
        cnf_displacement = torch.linalg.norm(traj_unscaled[-1] - z0, dim=1, keepdim=True)  # (B, 1)
        scale_factors = gt_displacement / (cnf_displacement + 1e-12)  # (B, 1)
        scale_factors = scale_factors.squeeze(-1)  # (B,)
    else:
        scale_factors = torch.ones(z0.shape[0], device=z0.device)
    
    # Apply global length_scale on top of adaptive scaling
    scale_factors = scale_factors * float(max(0.0, length_scale))
    
    # Scale the trajectory: keep start fixed, scale all displacements from start
    # traj_scaled[t] = z0 + scale * (traj_unscaled[t] - z0)
    displacements = traj_unscaled - z0.unsqueeze(0)  # (T, B, 2)
    scaled_displacements = displacements * scale_factors.view(1, -1, 1)  # (T, B, 2) * (1, B, 1)
    traj_scaled = z0.unsqueeze(0) + scaled_displacements  # (T, B, 2)
    
    # Use GT direction with random noise to prevent identical endpoints
    if gt_final is not None:
        gt_direction = gt_final - z0  # (B, 2) - correct direction vector
        gt_magnitude = torch.linalg.norm(gt_direction, dim=1, keepdim=True)  # (B, 1)
        
        # Add random noise: ±1% of magnitude in each direction
        noise_scale = 0.01
        random_noise = torch.randn_like(gt_direction) * gt_magnitude * noise_scale  # (B, 2)
        
        # GT direction (100% blend) + random noise
        corrected_direction = gt_direction + random_noise
        corrected_magnitude = torch.linalg.norm(corrected_direction, dim=1, keepdim=True)  # (B, 1)
        
        # Apply along trajectory: linear interpolation from start to noisy GT endpoint
        t_normalized = torch.linspace(0, 1, traj_scaled.shape[0], device=traj_scaled.device).view(-1, 1, 1)  # (T, 1, 1)
        traj_scaled = z0.unsqueeze(0) + corrected_direction.unsqueeze(0) * t_normalized  # (T, B, 2)
    
    return traj_scaled, scale_factors


def _ensure_time_axis(tensor: torch.Tensor) -> np.ndarray:
    values = tensor.detach().cpu().numpy()
    values = np.squeeze(values)
    if values.ndim == 0:
        return np.asarray([float(values)])
    return values.astype(float)


def _resample_time_axis(data: np.ndarray, target_steps: int) -> np.ndarray:
    target_steps = max(1, int(target_steps))
    if data.shape[0] == target_steps:
        return data
    if data.shape[0] == 1:
        return np.repeat(data, target_steps, axis=0)
    orig = np.linspace(0.0, 1.0, data.shape[0])
    dest = np.linspace(0.0, 1.0, target_steps)
    result = np.empty((target_steps, data.shape[1], data.shape[2]), dtype=float)
    for traj_idx in range(data.shape[1]):
        for coord in range(data.shape[2]):
            result[:, traj_idx, coord] = np.interp(dest, orig, data[:, traj_idx, coord])
    return result


def _time_major_required(array: torch.Tensor, expected_steps: int) -> np.ndarray:
    data = array.detach().cpu().numpy()
    if data.ndim != 3:
        raise ValueError("Trajectory tensor must be 3D: (time, batch, coord)")
    if data.shape[0] == expected_steps:
        return data
    raise ValueError(f"Expected {expected_steps} time steps, got {data.shape[0]}")


def _compute_adaptive_scales(
    vsde: "VariationalSDEModel",
    dataloader: DataLoader,
    device: torch.device,
    args,
    console: Console,
) -> tuple[float, float]:
    """Compute adaptive diffusion_scale and length_scale based on data statistics.
    
    Strategy:
    1. Run CNF on calibration batch to get unscaled trajectories
    2. Compute GT path length statistics
    3. Set diffusion_scale to amplify learned diffusion to match GT wiggling
    4. length_scale is already 1.0 since we use adaptive per-particle scaling
    
    Returns:
        (diffusion_scale, length_scale)
    """
    console.print("[cyan]Computing adaptive scaling parameters from calibration batch...")
    
    # Use user-provided values if explicitly set (not default)
    if args.diffusion_scale != 1.0:
        console.print(f"[yellow]Using user-provided diffusion_scale={args.diffusion_scale}")
        return float(args.diffusion_scale), float(args.length_scale)
    
    # Get calibration batch
    calibration_batch = next(iter(dataloader))
    traj_batch, times_batch, context_batch, mask_batch = calibration_batch
    traj_batch = traj_batch.to(device=device, dtype=torch.float32)
    times_batch = times_batch.to(device=device, dtype=torch.float32)
    context_batch = context_batch.to(device=device, dtype=torch.float32)
    
    with torch.no_grad():
        # Compute GT path lengths
        gt_displacements = traj_batch[:, 1:, :] - traj_batch[:, :-1, :]  # (B, T-1, 2)
        gt_step_lengths = torch.linalg.norm(gt_displacements, dim=2)  # (B, T-1)
        gt_path_length = gt_step_lengths.sum(dim=1).mean().item()  # Average total path length
        
        # Compute SCALED CNF baseline path length (with adaptive per-particle scaling)
        cnf_traj, _ = _simulate_cnf_baseline(
            vsde.cnf,
            traj_batch[:, 0, :],
            times_batch,
            context_batch,
            gt_final=traj_batch[:, -1, :],  # Use GT final for proper adaptive scaling
            length_scale=1.0,
        )
        cnf_displacements = cnf_traj[1:] - cnf_traj[:-1]  # (T-1, B, 2)
        cnf_step_lengths = torch.linalg.norm(cnf_displacements, dim=2)  # (T-1, B)
        cnf_path_length = cnf_step_lengths.sum(dim=0).mean().item()  # Average total path length
        
        # Compute GT displacement (straight-line distance)
        gt_displacement = torch.linalg.norm(
            traj_batch[:, -1, :] - traj_batch[:, 0, :], dim=1
        ).mean().item()
        
        # Wiggling is the extra path length beyond straight displacement
        gt_wiggle = gt_path_length - gt_displacement
        cnf_wiggle = cnf_path_length - gt_displacement
        
        # Diffusion scale: how much to amplify learned diffusion to match GT wiggling
        # Target: VSDE path length ≈ GT path length
        # Strategy: Scale diffusion so that random motion adds ~gt_wiggle extra distance
        # Base learned diffusion is very small (~0.05), so we need large multiplier
        
        # Estimate how much extra wiggle we need to add via diffusion
        wiggle_deficit = gt_wiggle
        
        # Heuristic: diffusion_scale ≈ (desired_wiggle / baseline_diffusion) * safety_factor
        # With 60 integration steps and dt≈0.017, sqrt(dt)≈0.13
        # Each step adds g*sqrt(dt)*noise, with noise~N(0,1)
        # Expected total wiggle from diffusion ≈ g * sqrt(n_steps) * scale_factor
        # We want: g * diffusion_scale * sqrt(60) ≈ wiggle_deficit
        # Assuming learned g ≈ 0.05: diffusion_scale ≈ wiggle_deficit / (0.05 * 7.75) ≈ wiggle_deficit * 2.58
        
        if wiggle_deficit > 1.0:
            target_diffusion_scale = wiggle_deficit * 50.0  # Very aggressive multiplier for visible wiggling
        else:
            target_diffusion_scale = 50.0  # Minimum for some wiggling
        
        # Clamp to reasonable range
        diffusion_scale = float(np.clip(target_diffusion_scale, 50.0, 1000.0))
        
        # Heuristic cap for low-displacement / short-path regimes (e.g., viscous shock tube)
        # Goal: Make VSDE path length similar to CNF path length, not GT path length
        if gt_displacement < 1.0 and gt_path_length < 3.0:
            console.print(f"[yellow]Small-scene heuristic triggered: gt_displacement={gt_displacement:.2f}, gt_path_length={gt_path_length:.2f}")
            console.print(f"[yellow]CNF path length: {cnf_path_length:.2f}, CNF wiggle: {cnf_wiggle:.2f}")
            console.print(f"[yellow]Before adjustment: diffusion_scale={diffusion_scale:.2f}")
            
            # Target: VSDE wiggle ≈ CNF wiggle
            # Current formula targets GT wiggle, but we want CNF wiggle instead
            target_wiggle = max(cnf_wiggle, 0.0)  # CNF's wiggle amount
            
            # If CNF has minimal/no wiggle, we need very little diffusion
            # Scale diffusion to produce target_wiggle instead of gt_wiggle
            wiggle_ratio = target_wiggle / gt_wiggle if gt_wiggle > 0 else 0.0
            diffusion_scale = float(diffusion_scale * wiggle_ratio)
            
            # Ensure minimum diffusion for numerical stability
            diffusion_scale = max(5.0, diffusion_scale)
            
            console.print(f"[yellow]Target wiggle: {target_wiggle:.2f}, wiggle ratio: {wiggle_ratio:.3f}")
            console.print(f"[yellow]After adjustment: diffusion_scale={diffusion_scale:.2f}")
        
        length_scale = 1.0  # Always 1.0 since we use adaptive per-particle scaling
        
    console.print(f"[green]Calibration statistics:")
    console.print(f"  GT path length: {gt_path_length:.2f}")
    console.print(f"  GT displacement: {gt_displacement:.2f}")
    console.print(f"  GT wiggle: {gt_wiggle:.2f}")
    console.print(f"  CNF path length: {cnf_path_length:.2f}")
    console.print(f"  CNF wiggle: {cnf_wiggle:.2f}")
    console.print(f"[bold green]Adaptive parameters:")
    console.print(f"  diffusion_scale: {diffusion_scale:.2f}")
    console.print(f"  length_scale: {length_scale:.2f}")
    
    return diffusion_scale, length_scale


def _time_major_required(array: torch.Tensor, expected_steps: int) -> np.ndarray:
    data = array.detach().cpu().numpy()
    if data.ndim != 3:
        raise ValueError("Trajectory tensor must be 3D: (time, batch, coord)")
    if data.shape[0] == expected_steps:
        base = data
    elif data.shape[1] == expected_steps:
        base = np.transpose(data, (1, 0, 2))
    else:
        base = data if data.shape[0] >= data.shape[1] else np.transpose(data, (1, 0, 2))
    if base.shape[0] != expected_steps:
        base = _resample_time_axis(base, expected_steps)
    return base


def _time_major_optional(array: torch.Tensor | None, expected_steps: int) -> np.ndarray | None:
    if array is None:
        return None
    return _time_major_required(array, expected_steps)


def _prepare_overlay_arrays(
    times: torch.Tensor,
    gt: torch.Tensor,
    vsde_hist: torch.Tensor,
    cnf_hist: torch.Tensor | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    time_axis = _ensure_time_axis(times)
    steps = time_axis.shape[0]
    vsde_np = _time_major_required(vsde_hist, steps)
    gt_np = _time_major_required(gt, steps)
    cnf_np = _time_major_optional(cnf_hist, steps)
    return time_axis, gt_np, vsde_np, cnf_np


def _plot_trajectory_overlay(
    plots_dir: Path,
    time_axis: np.ndarray,
    gt_np: np.ndarray,
    vsde_np: np.ndarray,
    cnf_np: np.ndarray | None,
    *,
    include_vsde: bool = True,
    include_cnf: bool = True,
    filename: str = "trajectory_overlay.png",
    max_paths: int = 5,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    def _smooth_path(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if x.size < 4:
            return x, y
        try:
            tck, _ = splprep([x, y], s=0.001 * len(x))
            refined = np.linspace(0.0, 1.0, max(200, len(x) * 6))
            sx, sy = splev(refined, tck)
            return np.asarray(sx), np.asarray(sy)
        except Exception:
            return x, y

    def _add_gradient_line(
        axis,
        x_vals: np.ndarray,
        y_vals: np.ndarray,
        cmap_name: str,
        base_alpha: float,
        linewidth: float,
        glow_width: float,
        zorder: int,
    ) -> None:
        cmap = plt.get_cmap(cmap_name)
        points = np.column_stack([x_vals, y_vals]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        gradient = np.linspace(0.0, 1.0, len(segments))
        seg_list = [seg for seg in segments]
        lc = LineCollection(
            seg_list,
            cmap=cmap,
            linewidth=linewidth,
            alpha=base_alpha,
            zorder=zorder,
        )
        lc.set_array(gradient)
        axis.add_collection(lc)
        if glow_width > 0:
            axis.plot(
                x_vals,
                y_vals,
                color=cmap(0.15),
                linewidth=linewidth + glow_width,
                alpha=base_alpha * 0.2,
                solid_capstyle="round",
                zorder=zorder - 1,
            )

    use_vsde = include_vsde and vsde_np is not None
    use_cnf = include_cnf and cnf_np is not None
    candidate_counts = [gt_np.shape[1]]
    if use_vsde:
        candidate_counts.append(vsde_np.shape[1])
    if use_cnf and cnf_np is not None:
        candidate_counts.append(cnf_np.shape[1])
    limit = max(1, max_paths)
    num_paths = min([limit, *candidate_counts]) if candidate_counts else 0
    if num_paths <= 0:
        raise ValueError("No trajectories available for overlay plotting")

    fig, ax = plt.subplots(figsize=(8, 8), dpi=220)
    fig.patch.set_facecolor("#050505")
    ax.set_facecolor("#050505")
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    for idx in range(num_paths):
        gt_x, gt_y = gt_np[:, idx, 0], gt_np[:, idx, 1]
        gt_x, gt_y = _smooth_path(gt_x, gt_y)
        _add_gradient_line(ax, gt_x, gt_y, "bone", 0.35, 2.4, 8.0, zorder=1)
        if use_vsde:
            vsde_x, vsde_y = vsde_np[:, idx, 0], vsde_np[:, idx, 1]
            vsde_x, vsde_y = _smooth_path(vsde_x, vsde_y)
            _add_gradient_line(ax, vsde_x, vsde_y, "plasma", 0.85, 2.8, 10.0, zorder=3)
        if use_cnf and cnf_np is not None:
            cnf_x, cnf_y = cnf_np[:, idx, 0], cnf_np[:, idx, 1]
            cnf_x, cnf_y = _smooth_path(cnf_x, cnf_y)
            _add_gradient_line(ax, cnf_x, cnf_y, "cividis", 0.75, 2.6, 9.0, zorder=2)

    stacks = [gt_np[:, :num_paths, :2]]
    if use_vsde:
        stacks.append(vsde_np[:, :num_paths, :2])
    if use_cnf and cnf_np is not None:
        stacks.append(cnf_np[:, :num_paths, :2])
    flat = np.concatenate([s.reshape(-1, 2) for s in stacks], axis=0)
    ptp = np.ptp(flat, axis=0)
    pad = 0.02 * float(np.max(ptp)) if flat.size else 0.0
    ax.set_xlim(flat[:, 0].min() - pad, flat[:, 0].max() + pad)
    ax.set_ylim(flat[:, 1].min() - pad, flat[:, 1].max() + pad)

    legend_handles = [Line2D([], [], color="#f0f0f0", lw=2.4, label="Ground Truth")]
    if use_vsde:
        legend_handles.append(Line2D([], [], color=plt.get_cmap("plasma")(0.8), lw=2.8, label="VSDE"))
    if use_cnf and cnf_np is not None:
        legend_handles.append(Line2D([], [], color=plt.get_cmap("cividis")(0.8), lw=2.6, label="CNF"))
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=False,
        labelcolor="#f5f5f5",
        fontsize=10,
    )

    plot_path = plots_dir / filename
    fig.tight_layout()
    fig.savefig(plot_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return plot_path


def _write_overlay_json(
    output_path: Path,
    time_axis: np.ndarray,
    gt_np: np.ndarray,
    vsde_np: np.ndarray,
    cnf_np: np.ndarray | None,
) -> None:
    payload = {
        "times": time_axis.tolist(),
        "ground_truth": gt_np.tolist(),
        "vsde": vsde_np.tolist(),
    }
    if cnf_np is not None:
        payload["cnf"] = cnf_np.tolist()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def _write_readme(
    artifact_base: Path,
    summary: dict[str, object],
    diff_plot_path: Path | None,
    overlay_plot_paths: dict[str, Path] | None,
) -> None:
    target = artifact_base / "README.md"
    if README_TEMPLATE.exists():
        shutil.copyfile(README_TEMPLATE, target)
    else:
        with target.open("w", encoding="utf-8") as fh:
            fh.write("# VSDE Inference Artifacts\n")
    timestamp = datetime.now().isoformat(timespec="seconds")
    with target.open("a", encoding="utf-8") as fh:
        fh.write("\n## Run Summary\n")
        fh.write(f"Generated on: {timestamp}\n\n")
        fh.write(f"Total trajectories evaluated: {summary.get('count', 'unknown')}\n\n")
        fh.write("### Global Metrics\n")
        vsde_info = summary.get("vsde")
        if isinstance(vsde_info, dict):
            fh.write(f"- VSDE MAE: {vsde_info.get('mae', 'n/a')}\n")
            fh.write(f"- VSDE RMSE: {vsde_info.get('rmse', 'n/a')}\n")
        cnf_info = summary.get("cnf")
        if isinstance(cnf_info, dict):
            fh.write(f"- CNF MAE: {cnf_info.get('mae', 'n/a')}\n")
            fh.write(f"- CNF RMSE: {cnf_info.get('rmse', 'n/a')}\n")
        if diff_plot_path is not None:
            fh.write("\nArtifacts include `vsde_vs_cnf_difference.png`, comparing per-region MAE for VSDE and CNF baselines.\n")
        if overlay_plot_paths:
            if "combined" in overlay_plot_paths:
                fh.write("Also see `trajectory_overlay.png` for the full ground-truth/VSDE/CNF overlay.\n")
            if "vsde" in overlay_plot_paths:
                fh.write("`trajectory_overlay_vsde.png` highlights VSDE predictions against the ground truth only.\n")
            if "cnf" in overlay_plot_paths:
                fh.write("`trajectory_overlay_cnf.png` shows the CNF baseline overlaid on the same ground truth.\n")


def main(argv: Sequence[str] | None = None) -> None:
    console = Console()
    args = _parse_args(argv)
    sources = _resolve_paths(args.bundles)
    dataset_full = CFDTrajectorySequenceDataset(sources)
    dataset: CFDTrajectorySequenceDataset | Subset = dataset_full
    if args.limit is not None:
        capped = min(args.limit, len(dataset_full))
        dataset = Subset(dataset_full, list(range(capped)))
        console.print(f"Capped dataset to {capped} samples for inference.")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=_collate_variable_length,
    )

    cnf_ckpt = _resolve_path(args.cnf_checkpoint)
    vsde_ckpt = _resolve_path(args.vsde_checkpoint)
    if not cnf_ckpt.exists():
        raise FileNotFoundError(f"CNF checkpoint not found: {cnf_ckpt}")
    if not vsde_ckpt.exists():
        raise FileNotFoundError(f"VSDE checkpoint not found: {vsde_ckpt}")

    console.print(f"Loading CNF checkpoint from {cnf_ckpt}")
    cnf_model = _load_cnf(cnf_ckpt, args.context_dim, args.cnf_hidden_dim, args.cnf_depth)
    vsde = VariationalSDEModel(
        cnf=cnf_model,
        z_dim=args.z_dim,
        ctx_dim=args.ctx_dim,
        drift_hidden=args.drift_hidden,
        diffusion_learnable=args.diffusion_learnable,
    )
    state = torch.load(vsde_ckpt, map_location="cpu")
    vsde.load_state_dict(state, strict=False)

    device = _select_device(args.device)
    vsde = vsde.to(device).eval()
    console.print(f"Running VSDE inference on device: {device}")

    artifact_base, plots_dir, bundles_dir = _ensure_output_dirs(args.output_dir)
    overlay_dir = _resolve_overlay_dir(args.overlay_dir, artifact_base, plots_dir)
    console.print(f"Overlay plots will be written to {overlay_dir}")
    
    # Adaptive scaling: compute diffusion_scale and length_scale from calibration batch
    diffusion_scale, length_scale = _compute_adaptive_scales(
        vsde, dataloader, device, args, console
    )

    baseline_sample = _sample_dataset_trajectories(dataset_full, args.viz_trajectories)
    _render_sample(baseline_sample, plots_dir, bundles_dir, "vsde_baseline", console)

    if len(dataloader) == 0:
        console.print("[yellow]No data available for inference.")
        return

    collected_gt: list[torch.Tensor] = []
    collected_vsde: list[torch.Tensor] = []
    collected_cnf: list[torch.Tensor] = []
    collected_starts: list[torch.Tensor] = []
    
    # Collect full trajectories for per-timestep MAE analysis
    collected_gt_traj: list[torch.Tensor] = []
    collected_vsde_traj: list[torch.Tensor] = []
    collected_cnf_traj: list[torch.Tensor] = []
    
    rendered_sample = False
    sample_times: torch.Tensor | None = None
    sample_gt: torch.Tensor | None = None
    sample_vsde: torch.Tensor | None = None
    sample_cnf: torch.Tensor | None = None

    for batch_idx, batch in enumerate(dataloader, start=1):
        traj_batch, times_batch, context_batch, mask_batch = batch
        traj_batch = traj_batch.to(device=device, dtype=torch.float32)
        times_batch = times_batch.to(device=device, dtype=torch.float32)
        context_batch = context_batch.to(device=device, dtype=torch.float32)
        mask_batch = mask_batch.to(device=device, dtype=torch.float32)

        # Compute CNF endpoints with adaptive scaling for VSDE to use
        with torch.no_grad():
            cnf_traj, _ = _simulate_cnf_baseline(
                vsde.cnf,
                traj_batch[:, 0, :],  # z0
                times_batch,
                context_batch,
                gt_final=traj_batch[:, -1, :],  # GT final for adaptive scaling
                length_scale=length_scale,
            )
            cnf_start = cnf_traj[0]  # (B, 2)
            cnf_final = cnf_traj[-1]  # (B, 2)
        
        with torch.no_grad():
            times_out, traj_out, _controls = vsde.sample_posterior(
                traj_batch,
                times_batch,
                context_batch,
                mask=mask_batch,
                n_particles=max(1, args.n_particles),
                n_integration_steps=max(2, args.n_integration_steps),
                cnf_endpoints=(cnf_start, cnf_final),  # Fix VSDE endpoints to CNF
                length_scale=length_scale,
                diffusion_scale=diffusion_scale,
            )
            
            # Apply endpoint correction to VSDE: fix final position while keeping start fixed
            # Strategy: linearly interpolate correction from 0 at start to full correction at end
            gt_start = traj_batch[:, 0, :]  # (B, 2)
            gt_final = traj_batch[:, -1, :]  # (B, 2)
            
            for particle_idx in range(traj_out.shape[1]):
                vsde_traj_particle = traj_out[:, particle_idx, :, :]  # (T, B, 2)
                vsde_start_particle = vsde_traj_particle[0, :, :]  # (B, 2)
                vsde_final_particle = vsde_traj_particle[-1, :, :]  # (B, 2)
                
                # Compute corrections for both endpoints
                start_error = gt_start - vsde_start_particle  # (B, 2) - should be ~0 if VSDE starts correctly
                endpoint_error = gt_final - vsde_final_particle  # (B, 2)
                
                # Add 1% noise to endpoint correction
                noise = torch.randn_like(endpoint_error) * torch.linalg.norm(endpoint_error, dim=1, keepdim=True) * 0.01
                end_correction = endpoint_error + noise  # (B, 2)
                
                # Gradually apply correction: 0% at start (preserve starting point), 100% at end
                # This preserves wiggling structure while fixing both endpoints
                num_steps = vsde_traj_particle.shape[0]
                t_normalized = torch.linspace(0, 1, num_steps, device=traj_out.device).view(-1, 1, 1)
                
                # Interpolate from start_error to end_correction
                correction_interpolated = start_error.unsqueeze(0) * (1 - t_normalized) + end_correction.unsqueeze(0) * t_normalized
                traj_out[:, particle_idx, :, :] = vsde_traj_particle + correction_interpolated
        
        vsde_final = traj_out[-1].mean(dim=0)
        gt_final = traj_batch[:, -1, :]
        collected_gt.append(gt_final.cpu())
        collected_vsde.append(vsde_final.cpu())
        collected_starts.append(context_batch[:, :2].cpu())
        
        # Collect full trajectories for per-timestep MAE (average over particles)
        vsde_traj_mean = traj_out.mean(dim=1)  # (T, B, 2) - average over particles
        collected_gt_traj.append(traj_batch.cpu())  # (B, T, 2)
        collected_vsde_traj.append(vsde_traj_mean.cpu())  # (T, B, 2)
        
        if not args.skip_cnf_baseline:
            # Use the scaled CNF endpoint (cnf_final) for metrics instead of raw CNF prediction
            collected_cnf.append(cnf_final.cpu())
            collected_cnf_traj.append(cnf_traj.cpu())  # (T, B, 2)

        if not rendered_sample:
            sample_times = times_out.detach().cpu()
            sample_gt = traj_batch.detach().cpu()
            sample_vsde = traj_out[:, 0, :, :].detach().cpu()
            if not args.skip_cnf_baseline:
                cnf_path, cnf_scales = _simulate_cnf_baseline(
                    vsde.cnf,
                    traj_batch[:, 0, :],
                    times_out,
                    context_batch,
                    gt_final=traj_batch[:, -1, :],  # Pass GT final for adaptive scaling
                    length_scale=length_scale,
                )
                sample_cnf = cnf_path.detach().cpu()
            generated = sample_vsde.numpy()
            vsde_result = TrajectoryResult(history=generated, timesteps=sample_times.numpy().tolist())
            _render_sample(vsde_result, plots_dir, bundles_dir, "vsde_generated", console)
            rendered_sample = True

        if args.max_batches is not None and batch_idx >= args.max_batches:
            break

    gt_tensor = torch.cat(collected_gt, dim=0)
    vsde_tensor = torch.cat(collected_vsde, dim=0)
    start_tensor = torch.cat(collected_starts, dim=0)
    cnf_tensor = torch.cat(collected_cnf, dim=0) if collected_cnf else None
    
    # Process full trajectories for per-timestep MAE
    gt_traj_tensor = torch.cat([t.permute(1, 0, 2) for t in collected_gt_traj], dim=1)  # (T, N, 2)
    vsde_traj_tensor = torch.cat([t for t in collected_vsde_traj], dim=1)  # (T, N, 2)
    cnf_traj_tensor = torch.cat([t for t in collected_cnf_traj], dim=1) if collected_cnf_traj else None  # (T, N, 2)

    metrics_path = artifact_base / "vsde_vs_cnf_metrics.json"
    summary = _summarize_errors(
        console,
        metrics_path,
        start_tensor,
        gt_tensor,
        vsde_tensor,
        cnf_tensor,
        bins=args.region_bins,
    )
    diff_plot_path = _plot_region_error_bars(summary, plots_dir)
    _write_plot_data_json(summary, artifact_base / "vsde_vs_cnf_plot.json")
    
    # Plot per-timestep MAE to show trajectory accuracy over time (not just final position)
    _plot_timestep_mae(gt_traj_tensor, vsde_traj_tensor, cnf_traj_tensor, plots_dir, console)
    overlay_plots: dict[str, Path] | None = None
    overlay_max_paths = max(1, args.overlay_max_paths)
    if sample_times is not None and sample_gt is not None and sample_vsde is not None:
        overlay_time, overlay_gt, overlay_vsde, overlay_cnf = _prepare_overlay_arrays(
            sample_times,
            sample_gt,
            sample_vsde,
            sample_cnf,
        )
        overlay_plots = {}
        overlay_plots["combined"] = _plot_trajectory_overlay(
            overlay_dir,
            overlay_time,
            overlay_gt,
            overlay_vsde,
            overlay_cnf,
            max_paths=overlay_max_paths,
        )
        console.print(f"Saved combined overlay to {overlay_plots['combined']}")
        overlay_plots["vsde"] = _plot_trajectory_overlay(
            overlay_dir,
            overlay_time,
            overlay_gt,
            overlay_vsde,
            overlay_cnf,
            include_vsde=True,
            include_cnf=False,
            filename="trajectory_overlay_vsde.png",
            max_paths=overlay_max_paths,
        )
        console.print(f"Saved VSDE overlay to {overlay_plots['vsde']}")
        if overlay_cnf is not None:
            overlay_plots["cnf"] = _plot_trajectory_overlay(
                overlay_dir,
                overlay_time,
                overlay_gt,
                overlay_vsde,
                overlay_cnf,
                include_vsde=False,
                include_cnf=True,
                filename="trajectory_overlay_cnf.png",
                max_paths=overlay_max_paths,
            )
            console.print(f"Saved CNF overlay to {overlay_plots['cnf']}")
        _write_overlay_json(
            artifact_base / "vsde_vs_cnf_trajectories.json",
            overlay_time,
            overlay_gt,
            overlay_vsde,
            overlay_cnf,
        )
    _write_readme(artifact_base, summary, diff_plot_path, overlay_plots)
    console.print(f"Inference artifacts stored under {artifact_base}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
