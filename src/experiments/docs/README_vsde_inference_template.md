# VSDE Inference Artifact Guide

This README describes the files emitted by the VSDE inference experiment step.

## Contents

- `plots/vsde_baseline.png` – Reference visualization of ground-truth CFD trajectories sampled from the dataset.
- `plots/vsde_generated.png` – Visualization of trajectories sampled from the trained VSDE controller.
- `plots/vsde_vs_cnf_difference.png` – Bar chart comparing regional mean absolute error (MAE) for VSDE vs. CNF baselines.
- `plots/trajectory_overlay.png` – Combined overlay of ground-truth, VSDE, and CNF trajectories.
- `plots/trajectory_overlay_vsde.png` – Ground-truth vs VSDE overlay for focused inspection.
- `plots/trajectory_overlay_cnf.png` – Ground-truth vs CNF overlay for baseline comparisons.
- `bundles/vsde_baseline.npz` – Serialized bundle containing the baseline trajectories used for `vsde_baseline.png`.
- `bundles/vsde_generated.npz` – Serialized bundle containing the VSDE-generated trajectories used for `vsde_generated.png`.
- `vsde_vs_cnf_metrics.json` – Global and per-region summary statistics (MAE/RMSE/median) for VSDE and CNF predictions.
- `vsde_vs_cnf_plot.json` – JSON payload with the precise values that feed the bar chart so plots can be regenerated offline.
- `error_distributions.npz` – MAE arrays for VSDE (and CNF when available) so downstream scripts can recreate histograms and tail statistics.

> Overlay PNGs are written to the overlay directory configured for the experiment step (defaults to the `plots/` subfolder next to other artifacts).
Each README copied into an artifact directory will append a run-specific summary (timestamp, trajectory counts, and global metrics).
