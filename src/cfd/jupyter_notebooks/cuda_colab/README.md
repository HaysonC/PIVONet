# CUDA CFD Notebook Guide

## Purpose

This folder contains `cuda_cfd.ipynb`, a Colab-friendly notebook that:

- Clones the PyFR viscous shock tube benchmark
- Runs the solver on a CUDA-enabled runtime (e.g., Google Colab GPU)
- Exports every `.pyfrs` snapshot to VTU, filters out Density/Velocity arrays, and stores them as NumPy `.npy` files
- Bundles the NumPy artifacts into a zip so you can download or sync them to Google Drive—perfect when you do not want to heat up your Mac or wait for local simulations

## Why use this workflow?

- **No local CUDA hardware needed:** Colab provides free/low-cost GPUs.
- **Batch post-processing:** Automated export and filtering prevents manual VTU handling.
- **Portable artifacts:** Density/velocity NumPy arrays slot directly into downstream ML/analysis pipelines.

## Files

| File | Description |
| --- | --- |
| `cuda_cfd.ipynb` | Main notebook that installs PyFR, runs the CUDA case, exports VTU/NumPy data, and renders a quick PyVista preview. |
| `README.md` | (This file) Overview and usage tips for the Colab workflow. |

## Quick start (Colab GPU)

1. Open [Google Colab](https://colab.research.google.com/), upload `cuda_cfd.ipynb`, and set **Runtime → Change runtime type → GPU**.
2. Execute the cells sequentially:
   - **Setup:** Installs PyFR and clones `PyFR-Test-Cases`.
   - **Simulation:** Imports the mesh, converts it to `.pyfrm`, and runs the CUDA solver with the provided `.ini` file.
   - **Visualization deps:** Installs PyVista + Xvfb for headless rendering.
   - **Export pipeline:** Iterates through `.pyfrs` outputs, writes `density` / `velocity` `.npy` files, and zips them into `results_density_velocity.zip` under Google Drive.
   - **Preview:** Exports the most recent solution to VTU and renders a static PyVista image (`visual.png`).
3. Download the generated `results_density_velocity.zip` (and optional `visual.png`) from your Drive or Colab workspace.

## Customization tips

- **Different PyFR case:** Update `sim_dir` to another folder inside `PyFR-Test-Cases` and ensure the mesh/config names align.
- **Output location:** Adjust `root_out` or `drive_dest_dir` to match your Drive hierarchy.
- **Fields of interest:** Modify the `export_and_filter` helper to capture additional variables (e.g., pressure, temperature).

Run time will vary with the number of `.pyfrs` snapshots and Colab’s GPU allocation, but the workflow is still faster than running PyFR locally without CUDA acceleration. Enjoy the cooler laptop! :)
