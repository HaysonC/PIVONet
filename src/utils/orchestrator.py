"""YAML-driven experiment pipeline orchestrator."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .paths import project_root
from .console_gate import is_prompt_active


@dataclass(frozen=True)
class ExperimentStep:
    name: str
    description: str
    script: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentSpec:
    slug: str
    name: str
    description: str
    steps: list[ExperimentStep]
    source_path: Path


class ExperimentOrchestrator:
    """Loads YAML experiments and executes their steps sequentially."""

    def __init__(
        self,
        *,
        experiments_dir: Path | None = None,
        console: Console | None = None,
        progress_mode: str | None = None,
        step_progress_mode: str = "auto",
    ) -> None:
        self.console = console or Console()
        self.project_dir = project_root()
        self.experiments_dir = experiments_dir or (self.project_dir / "src" / "experiments")
        self.progress_mode = progress_mode
        self.step_progress_mode = step_progress_mode or "auto"
        self.device = self._detect_device()
        self._experiments = self._load_experiments()
        self._completed_step_durations: list[float] = []
        self._step_duration_history: dict[str, list[float]] = {}
        self._eta_bootstrap_factor = 1.5
        self._eta_bootstrap_min_add = 2.0
        self._prompt_state_dir = self.project_dir / "cache" / "runtime" / "prompt_states"
        self._prompt_state_dir.mkdir(parents=True, exist_ok=True)
        self._current_prompt_state_file: Path | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def list_experiments(self) -> list[ExperimentSpec]:
        return list(self._experiments.values())

    def run(self, slug: str, overrides: dict[str, Any] | None = None) -> None:
        spec = self._experiments.get(slug)
        if not spec:
            raise KeyError(f"Experiment '{slug}' not found. Available: {', '.join(self._experiments)}")
        spec_to_run = self._apply_overrides(spec, overrides or {})
        if overrides:
            formatted = ", ".join(f"{k}={v}" for k, v in overrides.items())
            self.console.print(f"[cyan]Applying CLI overrides:[/] {formatted}")
        self._run_experiment(spec_to_run)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_experiments(self) -> dict[str, ExperimentSpec]:
        experiments: dict[str, ExperimentSpec] = {}
        if not self.experiments_dir.exists():
            self.console.print(f"[yellow]Experiment directory {self.experiments_dir} does not exist yet.[/]")
            return experiments

        for path in sorted(self.experiments_dir.glob("*.yml")) + sorted(self.experiments_dir.glob("*.yaml")):
            with path.open("r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}
            slug = str(raw.get("slug") or path.stem)
            name = raw.get("name", slug.replace("-", " ").title())
            description = raw.get("description", "")
            steps = [
                ExperimentStep(
                    name=step.get("name", f"step-{idx+1}"),
                    description=step.get("description", ""),
                    script=step["script"],
                    params=step.get("params", {}) or {},
                )
                for idx, step in enumerate(raw.get("steps", []))
            ]
            experiments[slug] = ExperimentSpec(slug=slug, name=name, description=description, steps=steps, source_path=path)
        return experiments

    def _detect_device(self) -> str:
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                return f"CUDA ({name})"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "Apple MPS"
        except Exception:
            pass
        return "CPU"

    def _run_experiment(self, spec: ExperimentSpec) -> None:
        if not spec.steps:
            self.console.print(f"[yellow]Experiment '{spec.name}' has no steps to run.[/]")
            return

        header = Panel.fit(
            f"Running experiment: [bold]{spec.name}[/]\n{spec.description}\nSource: {spec.source_path}",
            title="Experiment Orchestrator",
            subtitle=f"Device: {self.device}",
            style="cyan",
        )
        self.console.print(header)

        total = len(spec.steps)
        start = time.perf_counter()
        for idx, step in enumerate(spec.steps, start=1):
            self._run_step(idx, total, step)
        elapsed = time.perf_counter() - start
        self.console.print(Panel.fit(f"✅ Finished '{spec.name}' in {elapsed:.2f}s", style="green"))

    def _apply_overrides(self, spec: ExperimentSpec, overrides: dict[str, Any]) -> ExperimentSpec:
        if not overrides:
            return spec
        new_steps: list[ExperimentStep] = []
        for step in spec.steps:
            params = dict(step.params)
            for raw_key, value in overrides.items():
                target = raw_key.split(".", 1)
                if len(target) == 2:
                    step_token, param_name = target
                    if not self._step_matches(step, step_token):
                        continue
                else:
                    param_name = target[0]
                params[param_name.replace("-", "_")] = value
            new_steps.append(ExperimentStep(name=step.name, description=step.description, script=step.script, params=params))
        return ExperimentSpec(
            slug=spec.slug,
            name=spec.name,
            description=spec.description,
            steps=new_steps,
            source_path=spec.source_path,
        )

    @staticmethod
    def _normalize_token(value: str) -> str:
        return value.strip().lower().replace(" ", "-").replace("_", "-")

    def _step_matches(self, step: ExperimentStep, token: str) -> bool:
        return self._normalize_token(step.name) == self._normalize_token(token)

    def _run_step(self, position: int, total: int, step: ExperimentStep) -> None:
        label = f"[{position}/{total}] Running '{step.name}' on {self.device}"
        self.console.print(Panel.fit(f"{label}\n{step.description}", title="Step", style="magenta"))

        script_path = self._resolve_script(step.script)
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found for step '{step.name}': {script_path}")

        command = [sys.executable, str(script_path), *self._flatten_params(step.params)]
        self.console.print(f"→ Executing: {' '.join(command)}")

        env = os.environ.copy()
        if self.progress_mode:
            env["FLOW_PROGRESS_MODE"] = self.progress_mode
        prompt_state_file = self._create_prompt_state_file(step)
        env["FLOW_PROMPT_STATE"] = str(prompt_state_file)
        self._current_prompt_state_file = prompt_state_file
        self._write_prompt_state(False)

        use_bar = self._use_step_progress_bar()
        step_start = time.perf_counter()
        process = subprocess.Popen(command, cwd=self.project_dir, env=env)
        try:
            if use_bar:
                self._monitor_step_with_bar(step.name, process)
            else:
                planned_epochs = self._extract_epoch_hint(step)
                self._monitor_step_with_logs(step.name, process, step_start, position, total, planned_epochs)
        except Exception:
            process.kill()
            raise
        finally:
            self._cleanup_prompt_state_file()

        step_elapsed = time.perf_counter() - step_start
        if step_elapsed > 0:
            self._completed_step_durations.append(step_elapsed)
            self._step_duration_history.setdefault(step.name, []).append(step_elapsed)
        self.console.print(f"[green]✓[/] Step '{step.name}' completed in {step_elapsed:.1f}s.")

    @staticmethod
    def _extract_epoch_hint(step: ExperimentStep) -> int | None:
        epoch_keys = ("epochs", "num_epochs", "n_epochs", "max_epochs")
        for key in epoch_keys:
            value = step.params.get(key)
            if value is None:
                continue
            try:
                epoch_int = int(value)
            except (TypeError, ValueError):
                continue
            if epoch_int > 0:
                return epoch_int
        return None

    def _resolve_script(self, script: str) -> Path:
        script_path = Path(script)
        if script_path.is_absolute():
            return script_path
        if script_path.parts and script_path.parts[0] == "scripts":
            return self.experiments_dir / script_path
        return (self.project_dir / script_path).resolve()

    @staticmethod
    def _flatten_params(params: dict[str, Any]) -> list[str]:
        args: list[str] = []
        for key, value in params.items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    args.append(flag)
            elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
                for item in value:
                    args.extend([flag, str(item)])
            else:
                args.extend([flag, str(value)])
        return args

    def _use_step_progress_bar(self) -> bool:
        mode = (self.step_progress_mode or "auto").lower()
        if mode == "bar":
            return True
        if mode == "plain":
            return False
        # auto: prefer plain logging to avoid overlapping with child Rich bars
        return False

    def _monitor_step_with_bar(self, step_name: str, process: subprocess.Popen[Any]) -> None:
        with Progress(
            SpinnerColumn(),
            BarColumn(),
            TextColumn("{task.description}"),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task_id = progress.add_task(f"{step_name}...", total=100)
            while True:
                retcode = process.poll()
                if retcode is not None:
                    if retcode != 0:
                        raise subprocess.CalledProcessError(retcode, process.args)
                    progress.update(task_id, completed=100)
                    break
                progress.advance(task_id, 5)
                if progress.tasks[task_id].completed >= 100:
                    progress.update(task_id, completed=0)
                time.sleep(0.2)

    def _monitor_step_with_logs(
        self,
        step_name: str,
        process: subprocess.Popen[Any],
        start_time: float,
        position: int,
        total_steps: int,
        planned_epochs: int | None = None,
    ) -> None:
        poll_interval = 0.5
        log_interval = 2.5
        deferred_message: str | None = None
        last_log_at: float | None = None
        while True:
            retcode = process.poll()
            if retcode is not None:
                if retcode != 0:
                    raise subprocess.CalledProcessError(retcode, process.args)
                if deferred_message and not self._prompt_blocked():
                    self.console.log(deferred_message, markup=False)
                break
            elapsed = time.perf_counter() - start_time
            eta = self._estimate_step_eta(step_name, position, total_steps, elapsed)
            prefix = f"[{position}/{total_steps} {step_name}]"
            message = f"{prefix} running… {elapsed:.1f}s elapsed"
            remaining_steps = max(total_steps - position, 0)
            if planned_epochs:
                message += f" | {planned_epochs} epochs planned"
            if remaining_steps:
                suffix = "step" if remaining_steps == 1 else "steps"
                message += f" | {remaining_steps} {suffix} remaining"
            if eta is not None:
                message += f" | ETA {self._format_duration(eta)}"
            if self._prompt_blocked():
                deferred_message = message
                time.sleep(poll_interval)
                continue
            now = time.perf_counter()
            if deferred_message is not None:
                self.console.log(deferred_message, markup=False)
                deferred_message = None
                last_log_at = now
            elif last_log_at is None or (now - last_log_at) >= log_interval:
                self.console.log(message, markup=False)
                last_log_at = now
            time.sleep(poll_interval)

    def _estimate_step_eta(self, step_name: str, position: int, total_steps: int, elapsed: float) -> float | None:
        if total_steps <= 0:
            return None
        safe_position = min(max(position, 1), total_steps)
        remaining_steps = max(total_steps - safe_position, 0)
        avg_current = self._average_duration(self._step_duration_history.get(step_name))
        if avg_current is None:
            avg_current = self._average_duration(self._completed_step_durations)
        if avg_current is None:
            avg_current = self._bootstrap_current_duration(elapsed)

        if avg_current is None:
            return None

        avg_current = max(avg_current, elapsed)
        min_tail = elapsed * 0.1 if elapsed > 0 else 0.0
        current_remaining = max(avg_current - elapsed, min_tail)

        future_avg = self._average_duration(self._completed_step_durations)
        if future_avg is None:
            future_avg = avg_current
        future_remaining = remaining_steps * future_avg
        return current_remaining + future_remaining

    def _bootstrap_current_duration(self, elapsed: float) -> float | None:
        if elapsed <= 0:
            return None
        estimate = elapsed * self._eta_bootstrap_factor
        return max(estimate, elapsed + self._eta_bootstrap_min_add)

    @staticmethod
    def _average_duration(values: Sequence[float] | None) -> float | None:
        if not values:
            return None
        filtered = [v for v in values if v > 0]
        if not filtered:
            return None
        return sum(filtered) / len(filtered)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0.0, seconds)
        hours, remainder = divmod(int(seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours:d}h {minutes:02d}m"
        if minutes:
            return f"{minutes:d}m {secs:02d}s"
        return f"{secs:d}s"

    def _prompt_blocked(self) -> bool:
        return is_prompt_active() or self._child_prompt_active()

    def _child_prompt_active(self) -> bool:
        path = self._current_prompt_state_file
        if path is None:
            return False
        try:
            contents = path.read_text(encoding="utf-8").strip()
        except OSError:
            return False
        return contents == "1"

    def _create_prompt_state_file(self, step: ExperimentStep) -> Path:
        timestamp = int(time.time() * 1000)
        filename = f"{self._normalize_token(step.name)}_{timestamp}.state"
        path = self._prompt_state_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("0", encoding="utf-8")
        return path

    def _write_prompt_state(self, active: bool) -> None:
        if self._current_prompt_state_file is None:
            return
        try:
            self._current_prompt_state_file.write_text("1" if active else "0", encoding="utf-8")
        except OSError:
            return

    def _cleanup_prompt_state_file(self) -> None:
        path = self._current_prompt_state_file
        self._current_prompt_state_file = None
        if path is None:
            return
        try:
            path.unlink(missing_ok=True)
        except OSError:
            return


def _parse_overrides(pairs: Sequence[str] | None) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if not pairs:
        return overrides
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid override '{pair}'. Expected format step.param=value")
        key, raw_value = pair.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("Override key cannot be empty.")
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as exc:  # pragma: no cover - rare parsing failure
            raise ValueError(f"Could not parse override value for '{key}': {exc}") from exc
        overrides[key] = value
    return overrides


def _cli(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Manage Flow experiment pipelines.")
    parser.add_argument("--list", action="store_true", help="Print all available experiments.")
    parser.add_argument("--run", metavar="SLUG", help="Run the experiment matching the provided slug.")
    parser.add_argument("--dir", type=str, default=None, help="Override the experiments directory.")
    parser.add_argument(
        "--progress-mode",
        choices=("auto", "bars", "plain"),
        default=None,
        help="Force a training progress style for all experiment steps.",
    )
    parser.add_argument(
        "--step-progress",
        choices=("auto", "bar", "plain"),
        default="auto",
        help="Display style for per-step orchestration progress bars (auto defaults to plain logging).",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="STEP.PARAM=VALUE",
        help="Override step parameters at runtime (use 'step.param=value' or 'param=value').",
    )
    args = parser.parse_args(argv)

    experiments_dir = Path(args.dir) if args.dir else None
    orchestrator = ExperimentOrchestrator(
        experiments_dir=experiments_dir,
        progress_mode=args.progress_mode,
        step_progress_mode=args.step_progress,
    )
    overrides = _parse_overrides(args.overrides)

    if args.list:
        specs = orchestrator.list_experiments()
        if not specs:
            print("No experiments found.")
            return
        for spec in specs:
            print(f"- {spec.slug}: {spec.name} — {spec.description}")
    if args.run:
        orchestrator.run(args.run, overrides=overrides)


if __name__ == "__main__":
    _cli()
