"""Utility helpers for resolving project and data directories."""

from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_ROOT = _PROJECT_ROOT / "data"


def project_root() -> Path:
    """Return the absolute path to the project root directory."""

    return _PROJECT_ROOT


def data_root() -> Path:
    """Return the absolute path to the `data/` directory."""

    return _DATA_ROOT


def resolve_data_path(*relative_segments: str, create: bool = False) -> Path:
    """Resolve a path rooted at the repository ``data/`` directory.

    Args:
        *relative_segments: Individual path segments under ``data/``.
        create: If ``True``, create the parent directories when resolving.

    Returns:
        Absolute ``Path`` under the data directory.
    """

    path = _DATA_ROOT.joinpath(*relative_segments)
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path
