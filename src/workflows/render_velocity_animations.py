#!/usr/bin/env python3
"""Render velocity evolution animations for one or many CFD flows."""

from __future__ import annotations

import argparse
import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

try:  # Optional acceleration for magnitude calculations
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

from src.interfaces.data_sources import NpyVelocityFieldSource
from src.utils.paths import project_root

MESH_POINT_FILENAMES = ("mesh_points.npy", "points.npy")
DEFAULT_OUTPUT_NAME = "velocity_evolution.gif"
DEFAULT_PREVIEW_NAME = "velocity_first_frame.png"


@dataclass(frozen=True)
class FlowJob:
    slug: str
    velocity_dir: Path
    output_path: Path
    preview_path: Path | None


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--velocity-dir",
        default=None,
        help="Process a specific velocity snapshot directory (overrides auto-discovery).",
    )
    parser.add_argument(
        "--velocity-root",
        default="data",
        help="When --velocity-dir is absent, scan this root for flow folders containing velocity snapshots.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Explicit animation output path (requires --velocity-dir). Defaults to <output-root>/<slug>/velocity_evolution.gif.",
    )
    parser.add_argument(
        "--output-root",
        default="cache/artifacts",
        help="Base directory for automatically generated animation/preview artifacts.",
    )
    parser.add_argument(
        "--save-preview",
        action="store_true",
        help="Also export a PNG of the first frame alongside the animation for each flow.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Custom title text applied to every animation (overrides title-template).",
    )
    parser.add_argument(
        "--title-template",
        default="{flow} velocity evolution",
        help="Format string used when --title is omitted. Available tokens: {flow}, {slug}.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps"),
        default="auto",
        help="Device used for magnitude calculations (auto prefers MPS when available).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable Rich progress bars during rendering.",
    )
    parser.add_argument(
        "--force-progress",
        action="store_true",
        help="Render progress bars even when stdout is not attached to an interactive terminal.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=120,
        help="Maximum number of snapshots to animate (after temporal subsampling).",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Use every Nth snapshot when building the animation for faster previews.",
    )
    parser.add_argument(
        "--vector-stride",
        type=int,
        default=4,
        help="Spatial stride for quiver arrows (higher = fewer vectors).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=12,
        help="Frames per second for exported animations.",
    )
    parser.add_argument(
        "--extent",
        type=float,
        nargs=4,
        metavar=("xmin", "xmax", "ymin", "ymax"),
        default=(-2.0, 2.0, -2.0, 2.0),
        help="Spatial bounds for synthetic grid coordinates when no mesh points exist.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=None,
        help="Override inferred square grid dimension, e.g. 80 for 80x80 snapshots.",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=6.5,
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=6.0,
        help="Figure height in inches.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Figure DPI for rasterized outputs.",
    )
    parser.add_argument(
        "--cmap",
        default="plasma",
        help="Matplotlib colormap for quiver/scatter coloring.",
    )
    return parser.parse_args(argv)


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (project_root() / path).resolve()


def _load_mesh_points(directory: Path) -> np.ndarray | None:
    for filename in MESH_POINT_FILENAMES:
        candidate = directory / filename
        if candidate.exists():
            points = np.load(candidate, allow_pickle=False)
            array = np.asarray(points, dtype=np.float32)
            if array.ndim != 2 or array.shape[1] < 2:
                raise ValueError(
                    f"Mesh points in {candidate} must be shape (N, >=2), got {array.shape}."
                )
            return array[:, :2]
    return None


def _infer_flow_slug(velocity_dir: Path) -> str:
    name = velocity_dir.name
    if name.lower() == "velocity" and velocity_dir.parent.name:
        return velocity_dir.parent.name
    return name


def _humanize_slug(slug: str) -> str:
    tokens = slug.replace("_", " ").replace("-", " ").split()
    if not tokens:
        return slug
    return " ".join(
        token if token.isupper() else token.capitalize() for token in tokens
    )


def _format_title(args: argparse.Namespace, slug: str) -> str:
    if args.title:
        return args.title
    human = _humanize_slug(slug)
    try:
        return args.title_template.format(flow=human, slug=slug)
    except KeyError as err:
        raise ValueError(f"Unknown placeholder in title-template: {err}")


def _resolve_device_choice(console: Console, requested: str) -> str:
    device = requested.lower()
    if device == "cpu":
        return "cpu"
    if device == "mps":
        prefer_mps = True
    else:  # auto
        prefer_mps = True
    if not _TORCH_AVAILABLE:
        if device != "cpu":
            console.print(
                "[yellow]PyTorch is not installed; falling back to CPU for magnitude calculations.[/]"
            )
        return "cpu"
    assert torch is not None
    if (
        prefer_mps and torch.backends.mps.is_available()
    ):  # pragma: no cover - hardware specific
        console.print("[green]Using Apple MPS backend for magnitude calculations.[/]")
        return "mps"
    if device == "mps":
        console.print("[yellow]Apple MPS backend unavailable; using CPU instead.[/]")
    return "cpu"


def _load_snapshots(
    directory: Path,
    *,
    max_frames: int | None,
    frame_stride: int,
) -> tuple[list[float], list[np.ndarray]]:
    source = NpyVelocityFieldSource(directory)
    timesteps: list[float] = []
    fields: list[np.ndarray] = []
    stride = max(1, int(frame_stride))
    for index, (timestep, field) in enumerate(source.iter_velocity_fields()):
        if index % stride != 0:
            continue
        timesteps.append(float(timestep))
        fields.append(np.asarray(field, dtype=np.float32))
        if max_frames is not None and len(fields) >= max_frames:
            break
    return timesteps, fields


def _sanitize_field(field: np.ndarray) -> np.ndarray:
    array = np.asarray(field, dtype=np.float32)
    if array.ndim == 1:
        if array.size % 2 != 0:
            raise ValueError(
                "Velocity array with odd length cannot be reshaped into 2D vectors."
            )
        array = array.reshape(-1, 2)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError("Velocity field must be shape (N, >=2)")
    return array[:, :2]


def _infer_grid_size(field: np.ndarray) -> int | None:
    n_points = field.shape[0]
    root = int(round(math.sqrt(n_points)))
    if root * root == n_points:
        return root
    return None


def _prepare_grid_fields(
    planar_fields: Sequence[np.ndarray],
    *,
    grid_size: int | None,
) -> tuple[list[np.ndarray] | None, tuple[int, int] | None]:
    if not planar_fields:
        return None, None
    inferred = grid_size or _infer_grid_size(planar_fields[0])
    if inferred is None:
        return None, None
    try:
        reshaped = [field.reshape(inferred, inferred, 2) for field in planar_fields]
    except ValueError:
        return None, None
    return reshaped, (inferred, inferred)


def _subsample_grid(field: np.ndarray, stride: int) -> np.ndarray:
    stride = max(1, stride)
    return field[::stride, ::stride]


def _normalize_extents(coords: Iterable[float]) -> tuple[float, float, float, float]:
    values = tuple(float(v) for v in coords)
    if len(values) != 4:
        raise ValueError("Extent requires four floats: xmin xmax ymin ymax")
    xmin, xmax, ymin, ymax = values
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("Invalid extent bounds (min must be < max).")
    return xmin, xmax, ymin, ymax


def _build_writer(output_path: Path, fps: int) -> animation.AbstractMovieWriter:
    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        return animation.PillowWriter(fps=fps)
    return animation.FFMpegWriter(fps=fps)


def _compute_planar_magnitudes(
    planar_fields: Sequence[np.ndarray],
    *,
    device: str,
) -> list[np.ndarray]:
    if device != "mps":
        return [np.linalg.norm(field[:, :2], axis=1) for field in planar_fields]
    assert torch is not None
    mags: list[np.ndarray] = []
    torch_device = torch.device("mps")
    for field in planar_fields:
        tensor = torch.from_numpy(field[:, :2]).to(torch_device)
        mag = torch.linalg.norm(tensor, dim=1).cpu().numpy()
        mags.append(mag)
    return mags


def _compute_grid_magnitudes(
    grid_fields: Sequence[np.ndarray],
    *,
    device: str,
) -> list[np.ndarray]:
    if device != "mps":
        return [np.linalg.norm(field[..., :2], axis=2) for field in grid_fields]
    assert torch is not None
    mags: list[np.ndarray] = []
    torch_device = torch.device("mps")
    for field in grid_fields:
        tensor = torch.from_numpy(field[..., :2]).to(torch_device)
        mag = torch.linalg.norm(tensor, dim=2).cpu().numpy()
        mags.append(mag)
    return mags


def _animate_grid(
    *,
    timesteps: Sequence[float],
    grid_fields: Sequence[np.ndarray],
    output_path: Path,
    vector_stride: int,
    extent: tuple[float, float, float, float],
    figsize: tuple[float, float],
    dpi: int,
    fps: int,
    title: str,
    cmap: str,
    device: str,
    progress_callback: Callable[[int, int], None] | None,
) -> None:
    xmin, xmax, ymin, ymax = extent
    rows, cols = grid_fields[0].shape[:2]
    xs = np.linspace(xmin, xmax, cols)
    ys = np.linspace(ymin, ymax, rows)
    X, Y = np.meshgrid(xs, ys)

    stride = max(1, vector_stride)
    sampled_x = _subsample_grid(X, stride)
    sampled_y = _subsample_grid(Y, stride)

    mags = _compute_grid_magnitudes(grid_fields, device=device)
    vmin = min(float(m.min()) for m in mags)
    vmax = max(float(m.max()) for m in mags)
    norm = Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1e-6)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)

    initial = grid_fields[0]
    sampled_u = _subsample_grid(initial[..., 0], stride)
    sampled_v = _subsample_grid(initial[..., 1], stride)
    sampled_mag = _subsample_grid(mags[0], stride)

    quiv = ax.quiver(
        sampled_x,
        sampled_y,
        sampled_u,
        sampled_v,
        sampled_mag,
        cmap=cmap,
        norm=norm,
        scale_units="xy",
        scale=None,
        linewidth=0.4,
        alpha=0.9,
    )
    title_text = ax.set_title(f"{title}\n t = {timesteps[0]:.2f}", color="white")
    cbar = fig.colorbar(quiv, ax=ax, pad=0.02, label="|v|")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax, "yticklabels"), color="white")

    def _update(frame_index: int):
        field = grid_fields[frame_index]
        sampled_u = _subsample_grid(field[..., 0], stride)
        sampled_v = _subsample_grid(field[..., 1], stride)
        sampled_mag = _subsample_grid(mags[frame_index], stride)
        quiv.set_UVC(sampled_u, sampled_v, sampled_mag)
        title_text.set_text(f"{title}\n t = {timesteps[frame_index]:.2f}")
        return quiv, title_text

    interval_ms = max(1, int(1000 / max(1, fps)))
    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=len(grid_fields),
        interval=interval_ms,
        blit=False,
    )

    writer = _build_writer(output_path, fps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(output_path, writer=writer, dpi=dpi, progress_callback=progress_callback)
    plt.close(fig)


def _animate_unstructured(
    *,
    timesteps: Sequence[float],
    coords: np.ndarray,
    planar_fields: Sequence[np.ndarray],
    output_path: Path,
    vector_stride: int,
    figsize: tuple[float, float],
    dpi: int,
    fps: int,
    title: str,
    cmap: str,
    device: str,
    progress_callback: Callable[[int, int], None] | None,
) -> None:
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("Unstructured coordinates must be shape (N, >=2)")

    stride = max(1, vector_stride)
    indices = np.arange(0, coords.shape[0], stride)
    sampled_coords = coords[indices]

    mags = _compute_planar_magnitudes(planar_fields, device=device)
    vmin = min(float(m.min()) for m in mags)
    vmax = max(float(m.max()) for m in mags)
    norm = Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1e-6)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.tick_params(colors="white")
    ax.set_aspect("equal", adjustable="box")

    padding_x = 0.05 * (coords[:, 0].max() - coords[:, 0].min() or 1.0)
    padding_y = 0.05 * (coords[:, 1].max() - coords[:, 1].min() or 1.0)
    ax.set_xlim(coords[:, 0].min() - padding_x, coords[:, 0].max() + padding_x)
    ax.set_ylim(coords[:, 1].min() - padding_y, coords[:, 1].max() + padding_y)

    initial_field = planar_fields[0][indices]
    initial_mag = mags[0][indices]
    quiv = ax.quiver(
        sampled_coords[:, 0],
        sampled_coords[:, 1],
        initial_field[:, 0],
        initial_field[:, 1],
        initial_mag,
        cmap=cmap,
        norm=norm,
        scale_units="xy",
        scale=None,
        linewidth=0.4,
        alpha=0.9,
    )
    cbar = fig.colorbar(quiv, ax=ax, pad=0.02, label="|v|")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax, "yticklabels"), color="white")
    title_text = ax.set_title(f"{title}\n t = {timesteps[0]:.2f}", color="white")

    def _update(frame_index: int):
        field = planar_fields[frame_index][indices]
        quiv.set_UVC(field[:, 0], field[:, 1], mags[frame_index][indices])
        title_text.set_text(f"{title}\n t = {timesteps[frame_index]:.2f}")
        return quiv, title_text

    interval_ms = max(1, int(1000 / max(1, fps)))
    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=len(planar_fields),
        interval=interval_ms,
        blit=False,
    )

    writer = _build_writer(output_path, fps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(output_path, writer=writer, dpi=dpi, progress_callback=progress_callback)
    plt.close(fig)


def _animate_phase_space(
    *,
    timesteps: Sequence[float],
    planar_fields: Sequence[np.ndarray],
    output_path: Path,
    figsize: tuple[float, float],
    dpi: int,
    fps: int,
    title: str,
    cmap: str,
    device: str,
    progress_callback: Callable[[int, int], None] | None,
) -> None:
    mags = _compute_planar_magnitudes(planar_fields, device=device)
    vmin = min(float(m.min()) for m in mags)
    vmax = max(float(m.max()) for m in mags)
    norm = Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1e-6)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor("black")
    ax.tick_params(colors="white")
    ax.set_xlabel("vx", color="white")
    ax.set_ylabel("vy", color="white")
    title_text = ax.set_title(
        f"{title} (phase space)\n t = {timesteps[0]:.2f}", color="white"
    )
    scatter = ax.scatter(
        planar_fields[0][:, 0],
        planar_fields[0][:, 1],
        c=mags[0],
        cmap=cmap,
        norm=norm,
        s=8,
        alpha=0.7,
    )
    fig.colorbar(scatter, ax=ax, label="|v|")

    def _update(frame_index: int):
        scatter.set_offsets(planar_fields[frame_index][:, :2])
        scatter.set_array(mags[frame_index])
        title_text.set_text(f"{title} (phase space)\n t = {timesteps[frame_index]:.2f}")
        return (scatter,)

    interval_ms = max(1, int(1000 / max(1, fps)))
    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=len(planar_fields),
        interval=interval_ms,
        blit=False,
    )

    writer = _build_writer(output_path, fps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(output_path, writer=writer, dpi=dpi, progress_callback=progress_callback)
    plt.close(fig)


def _save_preview(
    preview_path: Path,
    *,
    grid_fields: Sequence[np.ndarray] | None,
    coords: np.ndarray | None,
    planar_fields: Sequence[np.ndarray],
    timesteps: Sequence[float],
    extent: tuple[float, float, float, float],
    figsize: tuple[float, float],
    dpi: int,
    cmap: str,
) -> None:
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    if grid_fields:
        rows, cols = grid_fields[0].shape[:2]
        xs = np.linspace(extent[0], extent[1], cols)
        ys = np.linspace(extent[2], extent[3], rows)
        X, Y = np.meshgrid(xs, ys)
        mag = np.linalg.norm(grid_fields[0], axis=2)
        im = ax.pcolormesh(X, Y, mag, shading="auto", cmap=cmap)
        ax.set_title(f"t = {timesteps[0]:.2f}")
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(im, ax=ax, label="|v|")
    elif coords is not None:
        mag = np.linalg.norm(planar_fields[0], axis=1)
        quiv = ax.quiver(
            coords[:, 0],
            coords[:, 1],
            planar_fields[0][:, 0],
            planar_fields[0][:, 1],
            mag,
            cmap=cmap,
            scale_units="xy",
            scale=None,
            linewidth=0.4,
            alpha=0.8,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"t = {timesteps[0]:.2f}")
        fig.colorbar(quiv, ax=ax, label="|v|")
    else:
        mag = np.linalg.norm(planar_fields[0], axis=1)
        scatter = ax.scatter(
            planar_fields[0][:, 0],
            planar_fields[0][:, 1],
            c=mag,
            cmap=cmap,
            s=8,
            alpha=0.6,
        )
        ax.set_title(f"Phase space preview t = {timesteps[0]:.2f}")
        fig.colorbar(scatter, ax=ax, label="|v|")
    fig.savefig(preview_path, bbox_inches="tight")
    plt.close(fig)


def _is_velocity_dir(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*.npy"))


def _discover_velocity_dirs(root: Path) -> list[Path]:
    candidates: list[Path] = []
    if _is_velocity_dir(root):
        candidates.append(root)
    velocity_subdirs = sorted(root.glob("*/velocity"))
    for subdir in velocity_subdirs:
        if _is_velocity_dir(subdir):
            candidates.append(subdir)
    return candidates


def _build_jobs(args: argparse.Namespace, *, console: Console) -> list[FlowJob]:
    jobs: list[FlowJob] = []
    output_root = _resolve_path(args.output_root)
    if args.velocity_dir:
        velocity_dir = _resolve_path(args.velocity_dir)
        if not _is_velocity_dir(velocity_dir):
            raise FileNotFoundError(f"No velocity snapshots found under {velocity_dir}")
        slug = _infer_flow_slug(velocity_dir)
        if args.output:
            output_path = _resolve_path(args.output)
        else:
            output_path = output_root / slug / DEFAULT_OUTPUT_NAME
        preview_path = None
        if args.save_preview:
            preview_path = output_path.with_name(DEFAULT_PREVIEW_NAME)
        jobs.append(
            FlowJob(
                slug=slug,
                velocity_dir=velocity_dir,
                output_path=output_path,
                preview_path=preview_path,
            )
        )
        return jobs

    velocity_root = _resolve_path(args.velocity_root)
    discovered = _discover_velocity_dirs(velocity_root)
    if not discovered:
        console.print(
            f"[yellow]No velocity directories found under {velocity_root}. Provide --velocity-dir to target a folder.[/]"
        )
        return jobs

    seen: dict[str, int] = {}
    for directory in discovered:
        slug = _infer_flow_slug(directory)
        counter = seen.get(slug, 0)
        seen[slug] = counter + 1
        unique_slug = slug if counter == 0 else f"{slug}-{counter + 1}"
        out_dir = output_root / unique_slug
        output_path = out_dir / DEFAULT_OUTPUT_NAME
        preview_path = out_dir / DEFAULT_PREVIEW_NAME if args.save_preview else None
        jobs.append(
            FlowJob(
                slug=unique_slug,
                velocity_dir=directory,
                output_path=output_path,
                preview_path=preview_path,
            )
        )
    return jobs


@contextmanager
def _progress_context(enabled: bool, console: Console):
    if not enabled:
        yield None
        return
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=None),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )
    with progress:
        yield progress


def _progress_callback(
    progress: Progress | None,
    task_id: TaskID | None,
) -> Callable[[int, int], None] | None:
    if progress is None or task_id is None:
        return None

    def _callback(frame_number: int, total_frames: int) -> None:
        total = total_frames or progress.tasks[task_id].total or 1
        completed = min(frame_number + 1, total)
        progress.update(task_id, total=total, completed=completed)

    return _callback


def _process_flow(
    job: FlowJob,
    args: argparse.Namespace,
    *,
    console: Console,
    device: str,
    progress: Progress | None,
) -> None:
    console.print(f"[cyan]Processing flow '{job.slug}' from[/] {job.velocity_dir}")

    timesteps, raw_fields = _load_snapshots(
        job.velocity_dir,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
    )
    if not raw_fields:
        console.print(
            f"[red]No velocity snapshots were loaded from {job.velocity_dir}.[/]"
        )
        return

    planar_fields = [_sanitize_field(field) for field in raw_fields]
    grid_fields, _ = _prepare_grid_fields(planar_fields, grid_size=args.grid_size)

    mesh_coords = None
    try:
        loaded_points = _load_mesh_points(job.velocity_dir)
        if loaded_points is not None:
            if loaded_points.shape[0] != planar_fields[0].shape[0]:
                console.print(
                    "[yellow]Mesh points exist but count does not match velocity vectors; ignoring.[/]"
                )
            else:
                mesh_coords = loaded_points
    except ValueError as err:
        console.print(f"[yellow]{err}[/]")

    console.print(
        f"Loaded [green]{len(planar_fields)}[/] snapshots spanning "
        f"t=[{timesteps[0]:.2f}, {timesteps[-1]:.2f}] from {job.velocity_dir}"
    )

    extent = _normalize_extents(args.extent)
    figsize = (args.fig_width, args.fig_height)
    title = _format_title(args, job.slug)

    if job.preview_path is not None:
        _save_preview(
            job.preview_path,
            grid_fields=grid_fields,
            coords=mesh_coords,
            planar_fields=planar_fields,
            timesteps=timesteps,
            extent=extent,
            figsize=figsize,
            dpi=args.dpi,
            cmap=args.cmap,
        )
        console.print(f"Preview frame saved to {job.preview_path}")

    frame_count = len(grid_fields) if grid_fields is not None else len(planar_fields)
    description = f"Encoding {job.slug} ({frame_count} frames)"
    task_id = None
    if progress is not None:
        task_id = progress.add_task(description, total=max(1, frame_count))
    progress_cb = _progress_callback(progress, task_id)

    if grid_fields is not None:
        _animate_grid(
            timesteps=timesteps,
            grid_fields=grid_fields,
            output_path=job.output_path,
            vector_stride=args.vector_stride,
            extent=extent,
            figsize=figsize,
            dpi=args.dpi,
            fps=args.fps,
            title=title,
            cmap=args.cmap,
            device=device,
            progress_callback=progress_cb,
        )
    elif mesh_coords is not None:
        _animate_unstructured(
            timesteps=timesteps,
            coords=mesh_coords,
            planar_fields=planar_fields,
            output_path=job.output_path,
            vector_stride=args.vector_stride,
            figsize=figsize,
            dpi=args.dpi,
            fps=args.fps,
            title=title,
            cmap=args.cmap,
            device=device,
            progress_callback=progress_cb,
        )
    else:
        console.print(
            "[yellow]Could not reshape snapshots onto a square grid. "
            "Rendering a velocity phase-space animation instead.[/]"
        )
        _animate_phase_space(
            timesteps=timesteps,
            planar_fields=planar_fields,
            output_path=job.output_path,
            figsize=figsize,
            dpi=args.dpi,
            fps=args.fps,
            title=title,
            cmap=args.cmap,
            device=device,
            progress_callback=progress_cb,
        )

    if progress is not None and task_id is not None:
        progress.update(task_id, completed=progress.tasks[task_id].total)
    console.print(f"[green]Animation written to[/] {job.output_path}")


def main(argv: Sequence[str] | None = None) -> None:
    console = Console()
    args = _parse_args(argv)
    if args.output and not args.velocity_dir:
        raise ValueError("--output requires --velocity-dir to target a single flow.")

    device = _resolve_device_choice(console, args.device)
    jobs = _build_jobs(args, console=console)
    if not jobs:
        return

    progress_enabled = not args.no_progress
    if progress_enabled and not console.is_terminal and not args.force_progress:
        console.print(
            "[yellow]Disabling live progress bars because the output stream doesn't support in-place updates. "
            "Pass --force-progress to override.[/]"
        )
        progress_enabled = False

    with _progress_context(progress_enabled, console) as progress:
        for job in jobs:
            _process_flow(job, args, console=console, device=device, progress=progress)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
