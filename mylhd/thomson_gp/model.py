"""High-level Thomson GP interface combining core computations and visualization."""

from __future__ import annotations

import numpy as np
from scipy import interpolate

from .core import ThomsonGPCore
from .utils import ThomsonGPUtils
from .visualize import ThomsonGPVisualizer


class ThomsonGP(ThomsonGPCore, ThomsonGPVisualizer, ThomsonGPUtils):
    """Facade class exposing both computational and visualization capabilities."""

    pass


__all__ = ["ThomsonGP"]
