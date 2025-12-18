"""High-level CFD pipeline that orchestrates PyFR, trajectories, and modeling."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import numpy as np

from .modeling import HybridModel
from .particle_trajectories import ParticleTrajectorySimulator
from .pyfr_simulation import run_pyfr_case
from ..interfaces.cfd import CFDSessionResult, ModelingResult, TrajectoryExport
from ..interfaces.config import SimulationConfig
from ..interfaces.data_sources import NpyVelocityFieldSource
from ..interfaces.pyfr import PyFRSimulationResult
from ..utils.cache import create_run_cache
from ..utils.config import load_config
from ..utils.paths import project_root, resolve_data_path


class CFDPipeline:
    """Orchestrates mesh import, simulation, and downstream modeling."""

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = config or load_config()
        self.cache_dir: Path | None = None

    def run(
        self,
        case_dir: Path | str | None = None,
        *,
        particles: int | None = None,
        max_steps: int | None = None,
        dt: float | None = None,
        run_name: str | None = None,
    ) -> CFDSessionResult:
        """Execute the CFD workflow from PyFR to exported trajectories."""
        case_path = self._resolve_case_dir(case_dir)
        self.cache_dir = create_run_cache(self.config.cache_subdir, run_name)
        simulation_result = run_pyfr_case(
            case_path,
            backend=self.config.pyfr_backend,
            vtus_dir=self._resolve_data_dir(
                self.config.pyfr_output_subdir, create=True
            ),
            density_dir=self.config.density_dir,
            velocity_dir=self.config.velocity_dir,
            points_dir=self._resolve_data_dir("cfd/npy/points", create=True),
        )

        trajectory_export = self._export_trajectories(
            simulation_result,
            num_particles=particles or self.config.trajectory_particles,
            max_steps=max_steps
            if max_steps is not None
            else self.config.trajectory_steps,
            dt=dt or self.config.trajectory_dt,
        )

        modeling = HybridModel(cache_dir=self.cache_dir).fit(trajectory_export)
        self._write_manifest(simulation_result, trajectory_export, modeling, run_name)

        cache_dir = self.cache_dir
        if cache_dir is None:
            raise RuntimeError("Cache directory missing after run")

        return CFDSessionResult(
            simulation=simulation_result,
            trajectory_export=trajectory_export,
            modeling=modeling,
            cache_dir=cache_dir,
        )

    def _resolve_case_dir(self, case_dir: Path | str | None) -> Path:
        if case_dir:
            resolved = Path(case_dir)
        elif self.config.pyfr_case:
            resolved = project_root() / self.config.pyfr_case
        else:
            raise ValueError("No PyFR case directory configured")
        if not resolved.exists():
            raise FileNotFoundError(f"PyFR case directory not found: {resolved}")
        return resolved.resolve()

    def _resolve_data_dir(self, subdir: str, create: bool = False) -> Path:
        segments = Path(subdir).parts
        path = resolve_data_path(*segments)
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def _export_trajectories(
        self,
        simulation: PyFRSimulationResult,
        num_particles: int,
        max_steps: int | None,
        dt: float,
    ) -> TrajectoryExport:
        source = NpyVelocityFieldSource(self.config.velocity_subdir)
        simulator = ParticleTrajectorySimulator(
            velocity_source=source,
            diffusion_coefficient=self.config.diffusion_constant,
            dt=dt,
        )
        trajectories = simulator.simulate(
            num_particles=num_particles, max_steps=max_steps
        )

        export_dir = self.config.trajectory_dir
        export_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        bundle_path = export_dir / f"trajectories_n{num_particles}_{stamp}.npz"
        np.savez(
            bundle_path,
            history=trajectories.history,
            timesteps=np.asarray(trajectories.timesteps),
        )

        return TrajectoryExport(
            path=bundle_path,
            num_particles=num_particles,
            steps=trajectories.history.shape[0] - 1,
            result=trajectories,
        )

    def _write_manifest(
        self,
        simulation: PyFRSimulationResult,
        trajectory: TrajectoryExport,
        modeling: ModelingResult,
        run_name: str | None,
    ) -> None:
        cache_dir = self.cache_dir
        if cache_dir is None:
            raise RuntimeError("Cache directory was not initialized for this run")
        manifest = {
            "run": run_name or cache_dir.name,
            "case": simulation.case_dir.name,
            "pyfr_backend": simulation.backend,
            "particles": trajectory.num_particles,
            "steps": trajectory.steps,
            "encoder": modeling.encoder_state.name if modeling.encoder_state else None,
            "cnf": modeling.cnf_state.name if modeling.cnf_state else None,
            "metrics": modeling.metrics,
            "cache_dir": str(cache_dir),
        }
        manifest_path = cache_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
