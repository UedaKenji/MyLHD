"""
mylhd - LHD Data Collection and Analysis Package

A comprehensive Python package for collecting and analyzing data from 
the Large Helical Device (LHD) plasma physics experiment.

Modules:
- anadata: Core module for LHD kaiseki data format parsing
- labcom_retrieve: Package for retrieving measurement data from LABCOM systems
- cts_utls: CTS analysis tools and utilities
"""

# Import main classes and functions for easy access
from . import anadata
from . import labcom_retrieve
from . import cts_utls
from . import thomson_gp

# Expose key classes at package level
from .anadata import KaisekiData
from .labcom_retrieve import LHDRetriever, LHDData
from .thomson_gp import ThomsonGP

__version__ = "0.1.0"
__author__ = "Kenji Ueda"
__email__ = "kenji.ueda@nifs.ac.jp"

__all__ = [
    "anadata",
    "labcom_retrieve", 
    "cts_utls",
    "thomson_gp",
    "KaisekiData",
    "LHDRetriever",
    "LHDData",
    "ThomsonGP",
]
