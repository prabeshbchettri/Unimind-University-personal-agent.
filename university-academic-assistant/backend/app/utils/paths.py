"""Filesystem path helpers.

Resolves the project root and the data directories so paths stay stable
regardless of the current working directory.
"""

from __future__ import annotations

from pathlib import Path

# backend/app/utils/paths.py -> backend/app -> backend -> project root
BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent


def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


def resolve_data_dir(name: str) -> Path:
    """Return ``PROJECT_ROOT/data/<name>``, creating it if missing."""
    directory = PROJECT_ROOT / "data" / name
    directory.mkdir(parents=True, exist_ok=True)
    return directory
