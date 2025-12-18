"""Standalone utility for rendering velocity field scatter + magnitude histograms."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from rich.console import Console

from src.utils.paths import project_root
from src.visualization.velocity_field import VelocityFieldPlotter


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="Velocity snapshot (.npy) to visualize.",
    )
    parser.add_argument(
        "--output",
        default="cache/artifacts/velocity_plots/velocity_phase.png",
        help="Where to store the rendered figure (PNG).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50_000,
        help="Velocity samples to draw for the scatter plot.",
    )
    parser.add_argument(
        "--cmap",
        default="viridis",
        help="Matplotlib colormap for the magnitude shading.",
    )
    parser.add_argument(
        "--fig-width", type=float, default=8.0, help="Figure width in inches."
    )
    parser.add_argument(
        "--fig-height", type=float, default=4.5, help="Figure height in inches."
    )
    parser.add_argument(
        "--seed", type=int, default=7, help="RNG seed for subsampling (optional)."
    )
    parser.add_argument(
        "--title", default=None, help="Optional title override for the scatter panel."
    )
    parser.add_argument(
        "--fallback-dir",
        default="data/cfd/npy/velocity",
        help="Directory to scan for the most recent snapshot when --input is missing or is itself a directory.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        default=True,
        help="Skip gracefully when no velocity snapshot can be resolved instead of raising an error.",
    )
    return parser.parse_args(argv)


def _resolve_path(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (project_root() / path).resolve()


def _collect_latest_snapshots(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if root.is_file():
        return [root]
    candidates = [p for p in root.glob("*.npy") if p.is_file()]
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)


def _discover_snapshot(
    primary: Path, fallback_dir: Path, *, allow_missing: bool, console: Console
) -> Path | None:
    candidates: list[Path] = []
    if primary.exists():
        candidates.extend(_collect_latest_snapshots(primary))
    else:
        parent = primary.parent if primary.parent != primary else primary
        pattern = primary.name
        if pattern and parent.exists():
            matches = [p for p in parent.glob(pattern) if p.is_file()]
            candidates.extend(
                sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)
            )
    if not candidates:
        candidates.extend(_collect_latest_snapshots(fallback_dir))
    if candidates:
        chosen = candidates[0]
        console.print(f"[cyan]Using velocity snapshot:[/] {chosen}")
        return chosen
    if allow_missing:
        console.print(
            f"[yellow]Velocity snapshot not found. Looked for {primary} and under {fallback_dir}. Skipping visualization step.[/]"
        )
        return None
    raise FileNotFoundError(f"Velocity snapshot not found: {primary}")


def main(argv: Sequence[str] | None = None) -> None:
    console = Console()
    args = _parse_args(argv)
    velocity_path = _resolve_path(args.input)
    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotter = VelocityFieldPlotter(
        sample_points=args.samples,
        cmap=args.cmap,
        figsize=(args.fig_width, args.fig_height),
        random_seed=args.seed,
    )
    fallback_dir = (
        _resolve_path(args.fallback_dir)
        if args.fallback_dir
        else _resolve_path("data/cfd/npy/velocity")
    )
    snapshot = _discover_snapshot(
        velocity_path, fallback_dir, allow_missing=args.allow_missing, console=console
    )
    if snapshot is None:
        return
    artifact = plotter.plot_from_file(
        snapshot, output_path=output_path, show=False, title=args.title
    )
    console.print(f"Velocity plot saved to {artifact.path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
