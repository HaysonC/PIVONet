"""Shared launch option definitions consumed by the CLI and Streamlit app."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class LaunchOptions:
    """Inputs that drive CFD workflows and visual tooling."""

    command: str
    case_dir: Path | None = None
    run_name: str | None = None
    particles: int = 256
    max_steps: int | None = None
    dt: float = 0.01
    diffusion_constant: float | None = None
    output_path: Path | None = None
    input_path: Path | None = None
    max_particles: int = 200
    velocity_samples: int = 50_000
    flow_overlay: bool = True
    extra_flags: tuple[str, ...] = field(default_factory=tuple)
    # Modeling hyperparameters
    model_input_path: Path | None = None
    model_run_name: str | None = None
    encoder_latent_dim: int = 16
    encoder_context_dim: int = 64
    encoder_steps: int = 8
    encoder_lr: float = 1e-3
    cnf_steps: int = 6
    cnf_lr: float = 2e-4
    cnf_hidden_dim: int = 128
    # Viewer options
    viewer_dataset: str | None = None

    def command_hint(self) -> str:
        short = f"flow-cli {self.command}"
        args = []
        if self.command == "import":
            args.extend(
                [
                    f"--particles {self.particles}",
                    f"--dt {self.dt}",
                ]
            )
            if self.max_steps:
                args.append(f"--steps {self.max_steps}")
            if self.run_name:
                args.append(f"--run-name {self.run_name}")
        elif self.command == "visualize" and self.input_path:
            args.append(f"--input {self.input_path}")
        elif self.command == "velocity" and self.input_path:
            args.append(f"--input {self.input_path}")
        elif self.command == "model" and self.model_input_path:
            args.append(f"--input {self.model_input_path}")
        if self.command == "viewer" and self.viewer_dataset:
            args.append(f"--dataset {self.viewer_dataset}")
        if self.command == "model":
            args.extend(
                [
                    f"--latent {self.encoder_latent_dim}",
                    f"--context {self.encoder_context_dim}",
                    f"--enc-steps {self.encoder_steps}",
                    f"--enc-lr {self.encoder_lr}",
                    f"--cnf-steps {self.cnf_steps}",
                    f"--cnf-lr {self.cnf_lr}",
                    f"--cnf-hidden {self.cnf_hidden_dim}",
                ]
            )
        args.extend(self.extra_flags)
        return " ".join([short, *args])

    @classmethod
    def conversational_commands(cls) -> Iterable[str]:
        return ("import", "visualize", "velocity", "model", "viewer")
