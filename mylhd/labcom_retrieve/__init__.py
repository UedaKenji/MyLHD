"""
LHD Data Retrieval Package

A Python package for retrieving LHD (Large Helical Device) measurement data
using the Retrieve.exe command-line tool on Windows systems.
"""

from .core import LHDRetriever, LHDData
from .utils import (
    setup_retrieve_path, 
    validate_retrieve_exe,
    check_windows_environment,
    get_default_retrieve_paths
)

# WSL utilities (optional import)
try:
    from .wsl_utils import (
        is_wsl,
        is_windows_compatible,
        find_windows_retrieve_exe,
        get_wsl_environment_info
    )
except ImportError:
    # WSL utilities not available - this is fine
    pass

__all__ = [
    "LHDRetriever",
    "LHDData", 
    "setup_retrieve_path",
    "validate_retrieve_exe",
    "check_windows_environment",
    "get_default_retrieve_paths"
]