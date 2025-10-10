"""High-level Thomson GP interface combining core computations and visualization."""

from __future__ import annotations

from scipy import interpolate
import numpy as np

from .core import ThomsonGPCore
from .visualize import ThomsonGPVisualizer
from .utils import ThomsonGPUtils



class ThomsonGP(ThomsonGPCore, ThomsonGPVisualizer, ThomsonGPUtils):
    """Facade class exposing both computational and visualization capabilities."""

    pass

__all__ = ["ThomsonGP"]
