"""Utilities for managing per-run cache state."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .paths import project_root


def cache_root() -> Path:
    """Return the repository cache directory, creating it if necessary."""

    root = project_root() / "cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_cache_path(*relative_segments: str, create: bool = False) -> Path:
    """Resolve a path under the cache root."""

    path = cache_root().joinpath(*relative_segments)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def create_run_cache(subdir: str | None = None, run_tag: str | None = None) -> Path:
    """Create a fresh cache directory for a specific run."""

    base = cache_root()
    if subdir:
        base = base.joinpath(*Path(subdir).parts)
    base.mkdir(parents=True, exist_ok=True)
    tag = run_tag or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    run_dir = base / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
