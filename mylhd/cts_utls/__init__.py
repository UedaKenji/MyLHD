"""
CTS Utilities Package

Collective Thomson Scattering (CTS) analysis tools and utilities for LHD data.
"""

from .CTSfosc_viewer import plot_all, plot_with_marginals, average_spectrum

__all__ = [
    "plot_all",
    "plot_with_marginals", 
    "average_spectrum"
]