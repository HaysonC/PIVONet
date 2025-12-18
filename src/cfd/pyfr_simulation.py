"""PyFR simulation helpers targeting the Metal backend."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np

from ..interfaces.pyfr import PyFRSimulationResult
from ..utils.paths import resolve_data_path

SOLUTION_TIME_RE = re.compile(r"-([0-9]+(?:\.[0-9]+)?)\.pyfrs$")


def run_pyfr_case(
    case_dir: Path,
    backend: str = "metal",
    vtus_dir: Path | None = None,
    density_dir: Path | None = None,
    velocity_dir: Path | None = None,
    points_dir: Path | None = None,
    force_import: bool = False,
) -> PyFRSimulationResult:
    """Run and export a PyFR case locally using the Metal backend.

    Args:
        case_dir: Directory containing the PyFR mesh, config, and solution files.
        backend: PyFR backend to pass to ``pyfr run`` (default: ``metal``).
        vtus_dir: Where to write temporary VTU exports. Defaults to ``case_dir/tmp``.
        density_dir: Where to save density numpy artifacts (default: ``data/cfd/npy/density``).
        velocity_dir: Where to save velocity numpy artifacts (default: ``data/cfd/npy/velocity``).
        points_dir: Where to save exported mesh point coordinates (default: ``data/cfd/npy/points``).
        force_import: Whether to re-run ``pyfr import`` even if the ``.pyfrm`` exists.

    Returns:
        A :class:`PyFRSimulationResult` describing the files that were written.
    """

    case_dir = case_dir.resolve()
    mesh_path = _discover_mesh(case_dir)
    cfg_path = _discover_config(case_dir)

    pyfrm_path = mesh_path.with_suffix(".pyfrm")
    vtus_dir = vtus_dir or case_dir / "tmp"
    density_dir = density_dir or resolve_data_path("cfd", "npy", "density")
    velocity_dir = velocity_dir or resolve_data_path("cfd", "npy", "velocity")
    points_dir = points_dir or resolve_data_path("cfd", "npy", "points")

    if force_import or not pyfrm_path.exists():
        _import_mesh(mesh_path, pyfrm_path, case_dir)

    _run_simulation(pyfrm_path, cfg_path, backend, case_dir)

    pyfrs = _collect_solution_files(case_dir)
    vtus = _export_vtu_files(mesh_path, pyfrs, vtus_dir)
    density_files: List[Path] = []
    velocity_files: List[Path] = []
    points_files: List[Path] = []

    for vtu_path in vtus:
        saved_density, saved_velocity, saved_points = _extract_fields_from_vtu(
            vtu_path, density_dir, velocity_dir, points_dir
        )
        density_files.extend(saved_density)
        velocity_files.extend(saved_velocity)
        if saved_points:
            points_files.append(saved_points)

    return PyFRSimulationResult(
        case_dir=case_dir,
        mesh_path=mesh_path,
        cfg_path=cfg_path,
        pyfrm_path=pyfrm_path,
        backend=backend,
        pyfrs=pyfrs,
        vtus=vtus,
        density_files=density_files,
        velocity_files=velocity_files,
        points_files=points_files,
    )


def _discover_mesh(case_dir: Path) -> Path:
    candidates = sorted(case_dir.glob("*.msh"))
    candidates.extend(sorted(case_dir.glob("*.msh.xz")))
    if not candidates:
        raise FileNotFoundError(f"No mesh (.msh) file found in {case_dir}")
    mesh_path = candidates[0]
    if mesh_path.suffix == ".xz":
        mesh_path = _decompress_mesh(mesh_path)
    return mesh_path


def _discover_config(case_dir: Path) -> Path:
    configs = sorted(case_dir.glob("*.ini"))
    if not configs:
        raise FileNotFoundError(f"No config (.ini) file found in {case_dir}")
    return configs[0]


def _decompress_mesh(compressed: Path) -> Path:
    uncompressed = compressed.with_suffix("")
    if uncompressed.exists():
        return uncompressed
    subprocess.run(["xz", "-dk", str(compressed)], check=True)
    if not uncompressed.exists():
        raise FileNotFoundError(f"Failed to decompress {compressed}")
    return uncompressed


def _import_mesh(mesh: Path, pyfrm: Path, cwd: Path) -> None:
    print(f"Importing mesh {mesh.name} -> {pyfrm.name}")
    subprocess.run(["pyfr", "import", str(mesh), str(pyfrm)], cwd=cwd, check=True)


def _run_simulation(pyfrm: Path, cfg: Path, backend: str, cwd: Path) -> None:
    print(f"Running PyFR {pyfrm.name} with backend {backend}")
    subprocess.run(
        ["pyfr", "run", "-b", backend, str(pyfrm), str(cfg)],
        cwd=cwd,
        check=True,
    )


def _collect_solution_files(case_dir: Path) -> List[Path]:
    candidates = list(case_dir.glob("*.pyfrs"))
    return sorted(candidates, key=_extract_solution_time)


def _extract_solution_time(path: Path) -> float:
    match = SOLUTION_TIME_RE.search(path.name)
    if match:
        return float(match.group(1))
    return 0.0


def _export_vtu_files(mesh: Path, pyfrs: Iterable[Path], vtus_dir: Path) -> List[Path]:
    vtus_dir.mkdir(parents=True, exist_ok=True)
    vtus: List[Path] = []
    for pyfrs_file in pyfrs:
        target = vtus_dir / f"{pyfrs_file.stem}.vtu"
        if target.exists():
            vtus.append(target)
            continue
        print(f"Exporting {pyfrs_file.name} -> {target.name}")
        subprocess.run(
            ["pyfr", "export", str(mesh), str(pyfrs_file), str(target)],
            cwd=mesh.parent,
            check=True,
        )
        vtus.append(target)
    return vtus


def _extract_fields_from_vtu(
    vtu_path: Path,
    density_dir: Path,
    velocity_dir: Path,
    points_dir: Path,
) -> tuple[List[Path], List[Path], Path | None]:
    pv = _require_pyvista()
    density_dir.mkdir(parents=True, exist_ok=True)
    velocity_dir.mkdir(parents=True, exist_ok=True)
    points_dir.mkdir(parents=True, exist_ok=True)

    mesh = pv.read(vtu_path)
    base = vtu_path.stem

    if "Density" not in mesh.point_data or "Velocity" not in mesh.point_data:
        raise KeyError(f"VTU {vtu_path} missing Density/Velocity fields")

    saved_density: List[Path] = []
    saved_velocity: List[Path] = []
    saved_points: Path | None = None

    density_path = density_dir / f"{base}_density.npy"
    np.save(density_path, np.asarray(mesh.point_data["Density"]))
    saved_density.append(density_path)

    velocity = np.asarray(mesh.point_data["Velocity"])
    if velocity.ndim == 2 and velocity.shape[1] >= 2:
        velocity = velocity[:, :2]
    velocity_path = velocity_dir / f"{base}_velocity.npy"
    np.save(velocity_path, velocity)
    saved_velocity.append(velocity_path)

    points = np.asarray(mesh.points)
    if points.ndim == 2 and points.shape[1] >= 2:
        points = points[:, :2]
    points_path = points_dir / f"{base}_points.npy"
    np.save(points_path, points)
    saved_points = points_path

    return saved_density, saved_velocity, saved_points


def _require_pyvista() -> Any:
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError(
            "pyvista is required to read VTU exports. Install it with `pip install pyvista`."
        ) from exc
    return pv
