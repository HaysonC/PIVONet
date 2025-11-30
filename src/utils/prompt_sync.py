"""Cross-process prompt coordination utilities.

When a child process presents an interactive prompt (questionary, input, etc.)
it can notify the orchestrator by writing to a shared state file. The parent
polls this file to pause verbose logs until the prompt is resolved.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Final

_STATE_ENV: Final[str] = "FLOW_PROMPT_STATE"


def _state_path() -> Path | None:
    value = os.getenv(_STATE_ENV)
    if not value:
        return None
    return Path(value)


def set_state_file(path: Path | None) -> None:
    """Set or clear the prompt state file for the current process."""

    if path is None:
        os.environ.pop(_STATE_ENV, None)
        return
    os.environ[_STATE_ENV] = str(path)


def mark_active(active: bool) -> None:
    """Mark the prompt as active/inactive by writing to the shared file."""

    path = _state_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("1" if active else "0", encoding="utf-8")
    except OSError:
        # Non-fatal; simply skip synchronization if the file system rejects the write.
        return


def read_active() -> bool:
    """Return True if the shared file reports an active prompt."""

    path = _state_path()
    if path is None:
        return False
    try:
        contents = path.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    return contents == "1"


@contextmanager
def external_prompt() -> None:
    """Context manager to set the shared prompt state around blocking input."""

    mark_active(True)
    try:
        yield
    finally:
        mark_active(False)
