"""
LHD Data Retrieval Package

A Python package for retrieving LHD (Large Helical Device) measurement data
using the Retrieve.exe command-line tool on Windows systems.
"""

from .core import LHDData, LHDRetriever
from .utils import (
    check_windows_environment,
    get_default_retrieve_paths,
    setup_retrieve_path,
    validate_retrieve_exe,
)

# WSL utilities (optional import)
try:
    from .wsl_utils import (
        find_windows_retrieve_exe,
        get_wsl_environment_info,
        is_windows_compatible,
        is_wsl,
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
    "get_default_retrieve_paths",
]
