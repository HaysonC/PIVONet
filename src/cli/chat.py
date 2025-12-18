"""Conversation-style helpers powered by Rich for the Flow CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rich import traceback
from rich.console import Console
from rich.panel import Panel

from ..interfaces.launch_options import LaunchOptions

traceback.install()


@dataclass
class FlowChat:
    GUIDE_PATH = Path("docs/flow_usage.md")

    console: Console = Console()
    _last_paths: dict[str, Path] = field(default_factory=dict)

    def _guide_note(self) -> str:
        return f"[bold white]Need a refresher?[/] {self.GUIDE_PATH.as_posix()}"

    def default_path(self, key: str) -> Path | None:
        return self._last_paths.get(key)

    def remember_path(self, key: str, path: Path | None) -> None:
        if path is not None:
            self._last_paths[key] = path

    def greet(self) -> None:
        banner = Panel(
            "[bold cyan]Hi there! I'm PIVO. Tell me what you'd like to do today.[/]",
            title="PIVO CLI",
            subtitle="(import | visualize | velocity | model | experiment | exit)",
        )
        self.console.print(banner)
        self.console.print(
            Panel(
                self._guide_note(),
                title="Flow Usage Guide",
                style="bright_blue",
            )
        )

    def say(self, message: str, style: str = "cyan") -> None:
        self.console.print(f"[bold {style}]→[/] {message}")

    def success(self, message: str) -> None:
        self.console.print(
            Panel(
                f"{message}\n\n{self._guide_note()}",
                title="✨ Success",
                style="green",
            )
        )

    def hint(self, option: LaunchOptions) -> None:
        self.console.print(
            Panel(
                f"[bold yellow]Try this command[/]: {option.command_hint()}\n\n{self._guide_note()}",
                title="Command Clue",
                subtitle="I can chat more if needed",
                style="bright_yellow",
            )
        )

    def wrap_error(self, error: Exception, option: LaunchOptions | None = None) -> None:
        self.console.print(
            Panel(
                f"[bold red]{error.__class__.__name__} happened:[/]\n{error}\n\n{self._guide_note()}",
                title="Oops",
                style="bright_red",
            )
        )
        if option:
            self.hint(option)
