"""Utilities that bridge CLI/GUI inputs with the diffusion encoder + CNF training stack."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..cfd.modeling import HybridModel
from ..interfaces.cfd import ModelingResult, TrajectoryExport
from ..interfaces.launch_options import LaunchOptions
from ..interfaces.modeling import ModelingConfig
from ..interfaces.trajectories import TrajectoryResult
from .cache import create_run_cache
from .trajectory_io import load_trajectory_bundle


def _build_export(bundle_path: Path, result: TrajectoryResult) -> TrajectoryExport:
	return TrajectoryExport(
		path=bundle_path,
		num_particles=result.history.shape[1],
		steps=result.history.shape[0] - 1,
		result=result,
	)


def training_cache_dir(config: ModelingConfig) -> Path:
	return create_run_cache(config.cache_subdir, config.run_name)


def modeling_config_from_options(options: LaunchOptions) -> ModelingConfig:
	run_name = options.model_run_name or (options.input_path.stem if options.input_path else None)
	return ModelingConfig(
		run_name=run_name,
		latent_dim=options.encoder_latent_dim,
		context_dim=options.encoder_context_dim,
		encoder_lr=options.encoder_lr,
		encoder_steps=options.encoder_steps,
		cnf_lr=options.cnf_lr,
		cnf_steps=options.cnf_steps,
		cnf_hidden_dim=options.cnf_hidden_dim,
	)


@dataclass
class TrainingOutcome:
	result: ModelingResult
	cache_dir: Path


def train_from_bundle(bundle_path: Path, config: ModelingConfig | None = None) -> TrainingOutcome:
	resolved = bundle_path.expanduser().resolve()
	if not resolved.exists():
		raise FileNotFoundError(f"Trajectory bundle not found: {resolved}")
	result = load_trajectory_bundle(str(resolved))
	export = _build_export(resolved, result)
	cfg = config or ModelingConfig(run_name=resolved.stem)
	cache_dir = training_cache_dir(cfg)
	hybrid = HybridModel(cache_dir=cache_dir, config=cfg)
	modeling_result = hybrid.fit(export)
	return TrainingOutcome(result=modeling_result, cache_dir=cache_dir)
