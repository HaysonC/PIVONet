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
    # Import flow-data options (repurposed legacy trajectory import)
    import_data_source: Path | None = None
    import_data_flow: str | None = None
    import_data_overwrite: bool = False
    # Import model options
    import_model_source: Path | None = None
    import_model_target: str | None = None
    import_model_overwrite: bool = False

    # Viewer options (deprecated)
    viewer_dataset: str | None = None

    def command_hint(self) -> str:
        short = f"flow-cli {self.command}"
        args = []
        if self.command == "import":
            if self.import_data_source:
                args.append(f"--source {self.import_data_source}")
            if self.import_data_flow:
                args.append(f"--flow {self.import_data_flow}")
            if self.import_data_overwrite:
                args.append("--overwrite")
        elif self.command == "visualize" and self.input_path:
            args.append(f"--input {self.input_path}")
        elif self.command == "model" and self.model_input_path:
            args.append(f"--input {self.model_input_path}")
        elif self.command == "import-model":
            if self.import_model_source:
                args.append(f"--source {self.import_model_source}")
            if self.import_model_target:
                args.append(f"--target {self.import_model_target}")
            if self.import_model_overwrite:
                args.append("--overwrite")
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
        # Streamlit UI uses this list; the conversational CLI uses its own menu.
        return ("import", "visualize", "velocity", "model")
