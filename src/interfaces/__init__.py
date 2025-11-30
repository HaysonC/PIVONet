# ./src/interfaces/__init__.py
"""Shared interfaces for the ODE/CFD toolchain."""

from .config import SimulationConfig
from .data_sources import NpyVelocityFieldSource, VelocityFieldSource
from .trajectories import TrajectoryResult
from .visualization import PlotArtifact, TrajectoryVisualizer
from .modeling import ModelingConfig

__all__ = [
	"SimulationConfig",
	"VelocityFieldSource",
	"NpyVelocityFieldSource",
	"TrajectoryResult",
	"TrajectoryVisualizer",
	"PlotArtifact",
	"ModelingConfig",
]
# This directory contains interface definitions for various components 
# It controls how top level parts for each module interact and understand each other