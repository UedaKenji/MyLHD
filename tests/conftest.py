"""Pytest configuration for the MyLHD project."""

from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    """Ensure the project source directory is available on sys.path."""
    root = Path(__file__).resolve().parent.parent
    src_dir = root / "src"
    if src_dir.is_dir():
        sys.path.insert(0, str(src_dir))
