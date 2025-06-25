"""
WSL (Windows Subsystem for Linux) utilities for LHD retrieve package.
This module provides WSL-specific functionality for accessing Windows tools from Linux.
"""

import os
import platform
import subprocess
from typing import Optional, List


def is_wsl() -> bool:
    """
    Check if running in WSL (Windows Subsystem for Linux).
    
    Returns:
        bool: True if running in WSL environment
    """
    try:
        # Check for WSL in kernel version
        with open('/proc/version', 'r') as f:
            version = f.read().lower()
            return 'microsoft' in version or 'wsl' in version
    except:
        return False


def is_windows_compatible() -> bool:
    """
    Check if the environment can run Windows executables.
    
    Returns:
        bool: True if Windows or WSL environment
    """
    return platform.system() == "Windows" or is_wsl()


def find_windows_retrieve_exe() -> Optional[str]:
    """
    Find Retrieve.exe in Windows paths accessible from WSL.
    
    Returns:
        str: Path to Retrieve.exe if found, None otherwise
    """
    if not is_wsl():
        return None
    
    # Common Windows paths accessible from WSL
    wsl_windows_paths = [
        "/mnt/c/LABCOM/Retrieve/bin/Retrieve.exe",
        "/mnt/c/LHD/Retrieve/Retrieve.exe", 
        "/mnt/c/Program Files/LHD/Retrieve/Retrieve.exe",
        "/mnt/c/Program Files (x86)/LHD/Retrieve/Retrieve.exe"
    ]
    
    for path in wsl_windows_paths:
        if os.path.exists(path):
            return path
    
    return None


def test_retrieve_exe(retrieve_path: str) -> bool:
    """
    Test if Retrieve.exe can be executed from WSL.
    
    Args:
        retrieve_path: Path to Retrieve.exe
        
    Returns:
        bool: True if executable and can run
    """
    if not os.path.exists(retrieve_path):
        return False
    
    try:
        # Try to run with help flag
        subprocess.run(
            [retrieve_path, "-h"],
            capture_output=True,
            timeout=10
        )
        # Return code might be non-zero for help, but should not timeout
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
        return False


def get_wsl_windows_paths() -> List[str]:
    """
    Get list of Windows paths accessible from WSL.
    
    Returns:
        List of potential Retrieve.exe paths in WSL
    """
    if not is_wsl():
        return []
    
    return [
        "/mnt/c/LABCOM/Retrieve/bin/Retrieve.exe",
        "/mnt/c/LHD/Retrieve/Retrieve.exe",
        "/mnt/c/Program Files/LHD/Retrieve/Retrieve.exe", 
        "/mnt/c/Program Files (x86)/LHD/Retrieve/Retrieve.exe"
    ]


def convert_wsl_path_to_windows(wsl_path: str) -> str:
    """
    Convert WSL path to Windows path.
    
    Args:
        wsl_path: Path in WSL format (/mnt/c/...)
        
    Returns:
        str: Windows format path (C:\...)
    """
    if wsl_path.startswith('/mnt/'):
        # Convert /mnt/c/path to C:\path
        parts = wsl_path.split('/')
        if len(parts) >= 3:
            drive = parts[2].upper() + ':'
            path_parts = parts[3:]
            return drive + '\\' + '\\'.join(path_parts)
    
    return wsl_path


def get_wsl_environment_info() -> dict:
    """
    Get WSL environment information.
    
    Returns:
        dict: WSL environment details
    """
    info = {
        'is_wsl': is_wsl(),
        'is_windows_compatible': is_windows_compatible(),
        'platform': platform.system(),
        'available_windows_paths': []
    }
    
    if is_wsl():
        wsl_paths = get_wsl_windows_paths()
        available_paths = [path for path in wsl_paths if os.path.exists(path)]
        info['available_windows_paths'] = available_paths
        
        # Check if we can access Windows C: drive
        info['windows_c_accessible'] = os.path.exists('/mnt/c/')
        
        # Try to find working Retrieve.exe
        retrieve_exe = find_windows_retrieve_exe()
        if retrieve_exe:
            info['retrieve_exe_found'] = retrieve_exe
            info['retrieve_exe_working'] = test_retrieve_exe(retrieve_exe)
    
    return info