"""
CTS Utilities Package

Collective Thomson Scattering (CTS) analysis tools and utilities for LHD data.
"""

from .CTSfosc_viewer import save_plot, plot_with_marginals, average_spectrum

__all__ = [
    "save_plot",
    "plot_with_marginals", 
    "average_spectrum"
]