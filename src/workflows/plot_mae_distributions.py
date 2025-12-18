"""Plot MAE error distributions for VSDE versus CNF across inference runs."""

from __future__ import annotations

import argparse
import json
from itertools import cycle
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dirs",
        nargs="+",
        required=True,
        help="Paths to VSDE inference artifact directories that contain error_distributions.npz.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Display names for each artifact directory (order must match --artifact-dirs).",
    )
    parser.add_argument(
        "--output-dir",
        default="cache/artifacts/moreplots",
        help="Directory where the summary JSON and plot should be written.",
    )
    parser.add_argument(
        "--figure-name",
        default="mae_error_distributions.png",
        help="Filename for the combined MAE distribution plot under the output directory's plots subfolder.",
    )
    parser.add_argument(
        "--summary-name",
        default="mae_distribution_summary.json",
        help="Filename for the JSON summary that accompanies the figure."
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=120,
        help="Number of histogram bins used for density estimation.",
    )
    return parser.parse_args()


def _load_error_arrays(artifact_dir: Path) -> tuple[np.ndarray, np.ndarray | None]:
    target = artifact_dir / "error_distributions.npz"
    if not target.exists():
        raise FileNotFoundError(f"Missing error distributions at {target}")
    with np.load(target) as bundle:
        vsde = bundle["vsde"].astype(float)
        cnf = bundle["cnf"].astype(float) if "cnf" in bundle else None
    return vsde, cnf


def _compute_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
    }


def _density_curve(values: np.ndarray, bins: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    hist, edges = np.histogram(values, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, hist


def main() -> None:
    args = _parse_args()
    if len(args.labels) != len(args.artifact_dirs):
        raise ValueError("--labels and --artifact-dirs must have the same length.")

    console = Console()
    runs: list[dict[str, object]] = []
    for label, raw_dir in zip(args.labels, args.artifact_dirs):
        artifact_dir = Path(raw_dir)
        vsde, cnf = _load_error_arrays(artifact_dir)
        entry = {
            "label": label,
            "count": int(len(vsde)),
            "vsde": vsde,
            "cnf": cnf,
            "stats": {"vsde": _compute_stats(vsde)} if len(vsde) else {"vsde": {}},
        }
        if cnf is not None:
            entry["stats"]["cnf"] = _compute_stats(cnf)
        runs.append(entry)

    all_errors = np.concatenate(
        [entry["vsde"] for entry in runs] +
        [entry["cnf"] for entry in runs if entry["cnf"] is not None],
        axis=0,
    ) if runs else np.array([])
    max_error = float(np.max(all_errors)) if all_errors.size else 1.0
    bins = np.linspace(0.0, max_error * 1.05 + 1e-6, max(3, args.bins))

    diff_arrays: list[np.ndarray] = []
    for entry in runs:
        cnf = entry.get("cnf")
        if cnf is not None:
            diff_arrays.append(cnf - entry["vsde"])
    diff_range = float(max(np.max(np.abs(arr)) for arr in diff_arrays)) if diff_arrays else max_error
    diff_bins = np.linspace(-diff_range * 1.1, diff_range * 1.1, max(3, args.bins))

    fig, axes = plt.subplots(1, 3, figsize=(19, 5), constrained_layout=True)
    palette = cycle([
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:teal",
    ])

    axes[0].set_title("ODE-only MAE (CNF)")
    axes[0].set_xlabel("Final-position MAE")
    axes[0].set_ylabel("Density")
    axes[1].set_title("VSDE vs ODE MAE (overlay)")
    axes[1].set_xlabel("Final-position MAE")
    axes[2].set_title("MAE difference (CNF − VSDE)")
    axes[2].set_xlabel("CNF MAE minus VSDE MAE")

    for entry in runs:
        color = next(palette)
        label = entry["label"]
        cnf_values = entry.get("cnf")
        if cnf_values is not None:
            centers, density = _density_curve(cnf_values, bins)
            axes[0].plot(centers, density, color=color, label=label, linewidth=2)
            centers_diff, density_diff = _density_curve(cnf_values - entry["vsde"], diff_bins)
            axes[2].plot(centers_diff, density_diff, color=color, linewidth=2, label=label)
        else:
            axes[0].plot([], [], color=color, label=label, linewidth=2)
        vsde_centers, vsde_density = _density_curve(entry["vsde"], bins)
        axes[1].plot(
            vsde_centers,
            vsde_density,
            color=color,
            linestyle="-",
            linewidth=2,
            label=f"{label} VSDE",
        )
        if cnf_values is not None:
            axes[1].plot(
                centers,
                density,
                color=color,
                linestyle="--",
                linewidth=2,
                label=f"{label} CNF",
            )
    axes[0].legend()
    axes[1].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    axes[2].axvline(0.0, color="#222", ls=":", lw=1.0)

    summary_lines: list[str] = []
    for entry in runs:
        parts = [f"{entry['label']} (N={entry['count']})"]
        cnf_stats = entry["stats"].get("cnf")
        if cnf_stats:
            parts.append(
                "CNF μ={mean:.3f} σ={std:.3f} median={median:.3f}".format(**cnf_stats)
            )
        vsde_stats = entry["stats"].get("vsde")
        if vsde_stats:
            parts.append(
                "VSDE μ={mean:.3f} σ={std:.3f} median={median:.3f}".format(**vsde_stats)
            )
        summary_lines.append(" | ".join(parts))
    for line in summary_lines:
        console.print(line)

    plots_dir = Path(args.output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    figure_path = plots_dir / args.figure_name
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)

    summary_payload = {
        "runs": [
            {
                "label": entry["label"],
                "count": entry["count"],
                "vsde": entry["stats"].get("vsde"),
                "cnf": entry["stats"].get("cnf"),
            }
            for entry in runs
        ],
        "plot": str(figure_path),
    }
    summary_path = Path(args.output_dir) / args.summary_name
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    console.print(f"Saved MAE distribution figure to {figure_path}")
    console.print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
