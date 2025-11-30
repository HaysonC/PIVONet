"""Global gate for coordinating console prompts and logging."""

from __future__ import annotations

from contextlib import contextmanager
from threading import Lock

from . import prompt_sync


class _PromptGate:
    def __init__(self) -> None:
        self._lock = Lock()
        self._depth = 0

    @contextmanager
    def activate(self):
        depth = self._increment()
        if depth == 1:
            prompt_sync.mark_active(True)
        try:
            yield
        finally:
            depth = self._decrement()
            if depth == 0:
                prompt_sync.mark_active(False)

    def _increment(self) -> int:
        with self._lock:
            self._depth += 1
            return self._depth

    def _decrement(self) -> int:
        with self._lock:
            self._depth = max(0, self._depth - 1)
            return self._depth

    def is_active(self) -> bool:
        with self._lock:
            return self._depth > 0


_prompt_gate = _PromptGate()


@contextmanager
def prompt_gate():
    """Context manager that marks a blocking prompt as active."""

    with _prompt_gate.activate():
        yield


def is_prompt_active() -> bool:
    """Return True when a blocking CLI prompt is currently active."""

    return _prompt_gate.is_active()
