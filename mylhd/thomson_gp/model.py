"""High-level Thomson GP interface combining core computations and visualization."""

from __future__ import annotations

from .core import ThomsonGPCore
from .visualize import ThomsonGPVisualizer


class ThomsonGP(ThomsonGPCore, ThomsonGPVisualizer):
    """Facade class exposing both computational and visualization capabilities."""

    pass


__all__ = ["ThomsonGP"]
