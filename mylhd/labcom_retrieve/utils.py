import os
import platform
import shutil
from pathlib import Path
from typing import Optional

try:
    from .wsl_utils import is_windows_compatible, find_windows_retrieve_exe
except ImportError:
    # Fallback if wsl_utils not available
    def is_windows_compatible():
        return platform.system() == "Windows"
    
    def find_windows_retrieve_exe():
        return None


def validate_retrieve_exe(retrieve_path: Optional[str] = None) -> bool:
    """
    Validate that Retrieve.exe exists and is accessible.
    
    Args:
        retrieve_path: Optional path to Retrieve.exe. If None, searches in PATH.
        
    Returns:
        bool: True if Retrieve.exe is found and accessible.
    """
    if not is_windows_compatible():
        raise RuntimeError("This package requires Windows or WSL environment")
    
    if retrieve_path:
        return os.path.isfile(retrieve_path) and os.access(retrieve_path, os.X_OK)
    
    return shutil.which("Retrieve.exe") is not None


def setup_retrieve_path(retrieve_dir: str) -> str:
    """
    Setup the path to Retrieve.exe directory.
    
    Args:
        retrieve_dir: Directory containing Retrieve.exe
        
    Returns:
        str: Full path to Retrieve.exe
        
    Raises:
        FileNotFoundError: If Retrieve.exe is not found in the specified directory
        RuntimeError: If not running on Windows
    """
    if not is_windows_compatible():
        raise RuntimeError("This package requires Windows or WSL environment")
    
    retrieve_path = Path(retrieve_dir) / "Retrieve.exe"
    
    if not retrieve_path.exists():
        raise FileNotFoundError(f"Retrieve.exe not found in {retrieve_dir}")
    
    if not os.access(retrieve_path, os.X_OK):
        raise PermissionError(f"Retrieve.exe is not executable: {retrieve_path}")
    
    return str(retrieve_path)


def get_default_retrieve_paths() -> list:
    """
    Get common default paths where Retrieve.exe might be installed.
    
    Returns:
        list: List of potential Retrieve.exe installation paths
    """
    common_paths = []
    
    if platform.system() == "Windows":
        # Native Windows paths
        common_paths = [
            r"C:\LABCOM\Retrieve\bin\Retrieve.exe",
            r"C:\LHD\Retrieve\Retrieve.exe",
            r"C:\Program Files\LHD\Retrieve\Retrieve.exe",
            r"C:\Program Files (x86)\LHD\Retrieve\Retrieve.exe",
            r".\Retrieve.exe",
            r".\bin\Retrieve.exe"
        ]
    else:
        # Try WSL paths
        wsl_retrieve = find_windows_retrieve_exe()
        if wsl_retrieve:
            common_paths = [wsl_retrieve]
        
        # Also check common WSL mount points
        common_paths.extend([
            "/mnt/c/LABCOM/Retrieve/bin/Retrieve.exe",
            "/mnt/c/LHD/Retrieve/Retrieve.exe",
            "/mnt/c/Program Files/LHD/Retrieve/Retrieve.exe",
            "/mnt/c/Program Files (x86)/LHD/Retrieve/Retrieve.exe"
        ])
    
    return [path for path in common_paths if os.path.exists(path)]


def check_windows_environment() -> dict:
    """
    Check Windows environment for LHD data retrieval compatibility.
    
    Returns:
        dict: Environment information including OS, architecture, and Retrieve.exe status
    """
    env_info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "architecture": platform.architecture()[0],
        "is_windows_compatible": is_windows_compatible(),
        "retrieve_in_path": shutil.which("Retrieve.exe") is not None,
        "default_paths_available": get_default_retrieve_paths()
    }
    
    # Add WSL-specific information if available
    try:
        from .wsl_utils import get_wsl_environment_info
        if platform.system() != "Windows":
            wsl_info = get_wsl_environment_info()
            env_info.update(wsl_info)
    except ImportError:
        pass
    
    return env_info