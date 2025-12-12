"""Generate comparison charts from integrator-sweep inference artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_METHODS = ["euler", "improved-euler", "rk4", "dopri5"]


class ComparisonRecord(NamedTuple):
    method: str
    vsde_mae: float | None
    cnf_mae: float | None


FILE_TEMPLATE = "vsde_vs_cnf_metrics.json"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        default="cache/artifacts/integrator-comparison",
        help="Base directory where integrator-specific artifacts live.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        action="extend",
        default=None,
        help="Names of method subdirectories to include in the comparison (can repeat flag).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Path to store generated charts (defaults to <artifact-root>/charts).",
    )
    return parser.parse_args(argv)


def _load_metrics(artifact_root: Path, method: str) -> ComparisonRecord | None:
    candidate = artifact_root / method / FILE_TEMPLATE
    if not candidate.exists():
        print(f"Warning: metrics file missing for '{method}' ({candidate}); skipping.")
        return None
    summary = json.loads(candidate.read_text(encoding="utf-8"))
    vsde_info = summary.get("vsde")
    cnf_info = summary.get("cnf")
    vsde_mae = _safe_float(vsde_info.get("mae")) if isinstance(vsde_info, dict) else None
    cnf_mae = _safe_float(cnf_info.get("mae")) if isinstance(cnf_info, dict) else None
    return ComparisonRecord(method=method, vsde_mae=vsde_mae, cnf_mae=cnf_mae)


def _safe_float(value: Any | None) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def plot_mae(records: Sequence[ComparisonRecord], output_path: Path) -> None:
    if not records:
        print("No records found; skipping MAE chart generation.")
        return
    methods = [rec.method for rec in records]
    indices = np.arange(len(records))
    width = 0.35
    vsde_maes = [rec.vsde_mae if rec.vsde_mae is not None else 0.0 for rec in records]
    cnf_maes = [rec.cnf_mae if rec.cnf_mae is not None else 0.0 for rec in records]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(indices - width / 2, vsde_maes, width, label="VSDE", color="#1f77b4")
    ax.bar(indices + width / 2, cnf_maes, width, label="CNF", color="#ff7f0e")

    ax.set_xticks(indices)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Final-position MAE")
    ax.set_title("VSDE vs CNF final-position MAE by integrator")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path)
    print(f"Saved MAE comparison chart to {output_path}")


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    artifact_root = Path(args.artifact_root)
    methods = args.methods or DEFAULT_METHODS
    output_base = Path(args.output_dir) if args.output_dir else (artifact_root / "charts")
    output_base.mkdir(parents=True, exist_ok=True)
    records: list[ComparisonRecord] = []
    for method in methods:
        record = _load_metrics(artifact_root, method)
        if record:
            records.append(record)
    summary_path = output_base / "comparison_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(
            [
                {"method": rec.method, "vsde_mae": rec.vsde_mae, "cnf_mae": rec.cnf_mae}
                for rec in records
            ],
            fh,
            indent=2,
        )
    print(f"Wrote comparison summary to {summary_path}")
    plot_mae(records, output_base / "comparison_mae.png")


if __name__ == "__main__":
    main()
