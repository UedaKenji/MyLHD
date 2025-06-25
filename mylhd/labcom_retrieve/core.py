import os
import subprocess
import tempfile
import warnings
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .utils import validate_retrieve_exe, setup_retrieve_path, get_default_retrieve_paths


@dataclass
class LHDData:
    """
    Container for LHD measurement data.
    
    Attributes:
        data: The measurement data as numpy array
        time: Time axis data
        metadata: Dictionary containing shot number, channel info, etc.
        units: Data units
        description: Data description
    """
    data: np.ndarray
    time: np.ndarray
    metadata: Dict[str, Any]
    units: str = ""
    description: str = ""
    
    def to_pandas(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame({
            'time': self.time,
            'data': self.data
        })
    
    def save_csv(self, filename: str) -> None:
        """Save data to CSV file."""
        df = self.to_pandas()
        df.to_csv(filename, index=False)
    
    def get_val(self) -> np.ndarray:
        """Convert raw data to voltage values using VResolution and VOffset from metadata."""
        #metadataにVResolutionかVCoefficient1があればその値を使う
        vresolution = None 
        if 'VResolution' in self.metadata:
            vresolution = self.metadata['VResolution']
        elif 'VCoefficient1' in self.metadata:
            vresolution = self.metadata['VCoefficient1']
        else:
            raise ValueError("VResolution or VCoefficient1 not found in metadata. Cannot convert to voltage.")
        #metadataにVOffsetあるいはVCoefficient0があればその値を使う
        voffset = None
        if 'VOffset' in self.metadata:
            voffset = self.metadata['VOffset']
        elif 'VCoefficient0' in self.metadata:
            voffset = self.metadata['VCoefficient0']
        else:
            voffset = 0.0   
        
        try:
            vresolution = float(vresolution)
            voffset = float(voffset)
        except (ValueError, TypeError):
            raise ValueError("VResolution or VOffset values are not numeric")
        
        if self.data.dtype == np.int8:
            # int8の場合は、int16に変換してから計算する
            val = self.data.astype(np.float32) * vresolution + voffset
        else:
            # int16やfloat32の場合はそのまま計算する
            val = self.data * vresolution + voffset

        return val
        
    
    @property
    def val(self) -> np.ndarray:
        """Get voltage values from data using VResolution and VOffset."""
        ## self._val を使うことで、get_val()を毎回呼び出す必要がなくなる
        if not hasattr(self, '_val'):
            self._val = self.get_val()
        # もし _val がまだ定義されていなければ、get_val() を呼び出して計算する
        # これにより、get_val() の計算が一度だけ行われ、以降はキャッシュされた値を返す
        return self._val
    
    # valの他にvoltageをエイリアス
    @property
    def voltage(self) -> np.ndarray:
        """Get voltage values from data using VResolution and VOffset."""
        return self.val

    def plot(self, **kwargs):
        """Plot the data using matplotlib."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting")
        
        plt.figure(figsize=kwargs.get('figsize', (10, 6)))
        plt.plot(self.time, self.data, **kwargs)
        plt.xlabel('Time')
        plt.ylabel(f'{self.description} [{self.units}]' if self.units else self.description)
        plt.title(f"Shot {self.metadata.get('shot', 'Unknown')}")
        plt.grid(True)
        plt.show()


class LHDRetriever:
    """
    Main class for retrieving LHD measurement data using Retrieve.exe.
    """
    
    def __init__(self, retrieve_path: Optional[str] = None, working_dir: Optional[str] = None):
        """
        Initialize LHD data retriever.
        
        Args:
            retrieve_path: Path to Retrieve.exe. If None, searches in PATH.
            working_dir: Working directory for temporary files.
        """
        if retrieve_path:
            self.retrieve_path = setup_retrieve_path(retrieve_path)
        else:
            # Check for available Retrieve.exe in default paths
            default_paths = get_default_retrieve_paths()
            if default_paths:
                self.retrieve_path = default_paths[0]  # Use first found path
            elif validate_retrieve_exe():
                self.retrieve_path = "Retrieve.exe"
            else:
                raise FileNotFoundError(
                    "Retrieve.exe not found. Please ensure it's installed or specify retrieve_path. "
                    "For WSL, it should be accessible at /mnt/c/LABCOM/Retrieve/bin/Retrieve.exe"
                )
        
        # Set working directory to Retrieve.exe directory by default
        if working_dir:
            self.working_dir = working_dir
        else:
            # Use the parent directory of Retrieve.exe as working directory
            import os
            self.working_dir = os.path.dirname(os.path.abspath(self.retrieve_path))
        
    def _run_retrieve(self, diag_name: str, shot_no: int, subshot_no: int, 
                     ch_no_name: int, file_prefix: Optional[str] = None, 
                     options: Optional[List[str]] = None) -> tuple:
        """
        Run Retrieve.exe with LHD-specific command format.
        Command: Retrieve DiagName ShotNo SubShotNo ChNo-Name [FileName] [options]
        
        Args:
            diag_name: Diagnostic name
            shot_no: Shot number
            subshot_no: Sub-shot number
            ch_no_name: Channel number or signal name
            file_prefix: Optional file name prefix
            options: Optional list of command options
            
        Returns:
            tuple: (dat_file, prm_file, time_file) paths
        """
        cmd = [self.retrieve_path, diag_name, str(shot_no), str(subshot_no), str(ch_no_name)]
        
        if file_prefix:
            cmd.append(file_prefix)
        
        if options:
            cmd.extend(options)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.working_dir,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Retrieve.exe failed: {result.stderr}\n"+
                                      f"Command: {' '.join(cmd)}\n"+
                                      f"cwd: {self.working_dir}")
                                   
            
            # Determine output files based on actual Retrieve.exe naming convention
            # Retrieve.exe creates files with pattern: prefix-shot-subshot-channel.ext
            if file_prefix:
                # Look for files that start with the prefix
                from pathlib import Path
                working_path = Path(self.working_dir)
                
                # Find actual generated files
                dat_files = list(working_path.glob(f"{file_prefix}*.dat"))
                prm_files = list(working_path.glob(f"{file_prefix}*.prm"))
                time_files = list(working_path.glob(f"{file_prefix}*.time"))
                
                if dat_files:
                    dat_file = str(dat_files[0])
                    # Derive other file names from dat file
                    base_path = dat_file[:-4]  # Remove .dat extension
                    prm_file = f"{base_path}.prm"
                    time_file = f"{base_path}.time"
                else:
                    # Fallback to expected names
                    dat_file = os.path.join(self.working_dir, f"{file_prefix}.dat")
                    prm_file = os.path.join(self.working_dir, f"{file_prefix}.prm")
                    time_file = os.path.join(self.working_dir, f"{file_prefix}.time")
            else:
                # Default naming when no prefix specified
                base_name = f"{diag_name}_{shot_no}_{subshot_no}_{ch_no_name}"
                dat_file = os.path.join(self.working_dir, f"{base_name}.dat")
                prm_file = os.path.join(self.working_dir, f"{base_name}.prm")
                time_file = os.path.join(self.working_dir, f"{base_name}.time")
            
            return dat_file, prm_file, time_file
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Retrieve.exe timeout after 5 minutes")
        except FileNotFoundError:
            raise FileNotFoundError(f"Retrieve.exe not found: {self.retrieve_path}")
    
    def create_example_retrieval(self, diag_name: str = "Mag", shot: int = 139400, 
                               subshot: int = 1, channel: int = 32) -> str:
        """
        Create example command for data retrieval.
        
        Args:
            diag_name: Diagnostic name (default: 'Mag') 
            shot: Shot number (default: 139400)
            subshot: Sub-shot number (default: 1)
            channel: Channel number (default: 32)
            
        Returns:
            Example command string
        """
        return f"Retrieve {diag_name} {shot} {subshot} {channel} -T"
    
    def retrieve_data(self, 
                     diag_name: str,
                     shot: int, 
                     subshot: int,
                     channel: int, 
                     time_axis: bool = False,
                     frame_number: Optional[int] = None,
                     dtype: Optional[Union[str, np.dtype]] = None) -> LHDData:
        """
        Retrieve measurement data using LHD Retrieve.exe format.
        
        Args:
            diag_name: Diagnostic name (e.g., 'Mag', 'Magnetics')
            shot: Shot number
            subshot: Sub-shot number (usually 1)
            channel: Channel number
            time_axis: Generate time axis information (-T option)
            frame_number: Specific frame number (-f option)
            dtype: Data type for reading binary data (str like 'float32', 'int8' or numpy dtype like np.float32, np.int8)
            
        Returns:
            LHDData object containing the retrieved data
        """
        # Prepare options
        options = []
        if time_axis:
            options.append('-T')
        if frame_number is not None:
            options.extend(['-f', str(frame_number)])
        
        # Generate unique file prefix
        file_prefix = f"retrieve_{diag_name}_{shot}_{subshot}_{channel}"
        
        try:
            dat_file, prm_file, time_file = self._run_retrieve(
                diag_name=diag_name,
                shot_no=shot,
                subshot_no=subshot,
                ch_no_name=channel,
                file_prefix=file_prefix,
                options=options
            )
            
            # Parse the output files
            data, time, metadata = self._parse_retrieve_files(dat_file, prm_file, time_file, dtype)
            
            return LHDData(
                data=data,
                time=time,
                metadata={
                    'diag_name': diag_name,
                    'shot': shot,
                    'subshot': subshot,
                    'channel': channel,
                    'time_axis': time_axis,
                    'frame_number': frame_number,
                    'dtype': dtype,
                    **metadata
                },
                description=f"{diag_name} Shot {shot}.{subshot}, Channel {channel}"
            )
            
        finally:
            # Clean up ALL temporary files generated by Retrieve.exe
            self._cleanup_temporary_files(file_prefix)
            pass
    
    def _cleanup_temporary_files(self, file_prefix: str) -> None:
        """
        Clean up all temporary files generated by Retrieve.exe.
        
        Args:
            file_prefix: Prefix used for temporary files
        """
        if not file_prefix:
            return
            
        try:
            from pathlib import Path
            working_path = Path(self.working_dir)
            
            # Find all files that start with the prefix
            # Retrieve.exe can generate various file extensions
            temp_file_patterns = [
                f"{file_prefix}*.dat",
                f"{file_prefix}*.prm", 
                f"{file_prefix}*.time",
                f"{file_prefix}*.tprm",
                f"{file_prefix}*.tmp"
            ]
            
            files_deleted = 0
            for pattern in temp_file_patterns:
                for temp_file in working_path.glob(pattern):
                    try:
                        temp_file.unlink()  # Delete file
                        files_deleted += 1
                    except (OSError, PermissionError):
                        # Continue even if some files can't be deleted
                        pass
            
            # Optional: log cleanup for debugging
            # print(f"Cleaned up {files_deleted} temporary files with prefix '{file_prefix}'")
            
        except Exception:
            # Don't let cleanup failures affect the main operation
            pass
    
    def _parse_retrieve_files(self, dat_file: str, prm_file: str, time_file: str, dtype: Optional[Union[str, np.dtype]] = None) -> tuple:
        """
        Parse output files from Retrieve.exe (.dat, .prm, .time).
        
        Args:
            dat_file: Path to .dat file (measurement data)
            prm_file: Path to .prm file (parameters)
            time_file: Path to .time file (time axis data)
            dtype: Data type for reading binary data (str or numpy dtype)
            
        Returns:
            Tuple of (data, time, metadata)
        """
        metadata = {}
        
        # Read parameter file (.prm)
        if os.path.exists(prm_file):
            try:
                #prm_fileはcsv形式で保存されており，２列目をキー，3列目を値として読み込む
                prm_data = pd.read_csv(prm_file, header=None, index_col=1)
                prm_data = prm_data.iloc[:, 1]  # Get second column as values
                metadata = prm_data.to_dict()  # Convert to dictionary
            except pd.errors.EmptyDataError:
                warnings.warn(f"Parameter file {prm_file} is empty or malformed.")
            except FileNotFoundError:
                warnings.warn(f"Parameter file not found: {prm_file}")

        # Read data file (.dat)
        if not os.path.exists(dat_file):
            raise FileNotFoundError(f"Data file not found: {dat_file}")
        
        try:
            # Choose data type based on dtype parameter
            if dtype is not None:
                # Handle both string and numpy dtype objects
                data = np.fromfile(dat_file, dtype=dtype)
            else:
                # For raw data, use int16
                data = np.fromfile(dat_file, dtype=np.int16)
                if len(data) == 0:
                    # Try int16 as fallback
                    data = np.fromfile(dat_file, dtype=np.int8).astype(np.int16)
                
        except Exception as e:
            # Fallback: try reading as text
            try:
                data = np.loadtxt(dat_file)
            except Exception as e2:
                raise RuntimeError(f"Failed to read data file {dat_file}: {e}, {e2}")
        
        # Read time file (.time) if available
        time_data = None
        if data.size > 10000000:
            print("Warning: Data size is very large, generating time axis is skipped.")
        else:
            if os.path.exists(time_file):
                try:
                    time_data = np.fromfile(time_file, dtype=np.float32)
                    if len(time_data) == 0:
                        time_data = np.fromfile(time_file, dtype=np.float64)
                except Exception as e:
                    warnings.warn(f"Failed to read time file {time_file}: {e}")
            
            # Generate time axis if not available
            if time_data is None or len(time_data) != len(data):
                # Use sampling rate from metadata if available
                sampling_rate = metadata.get('SamplingRate', metadata.get('sampling_rate', 1.0))
                try:
                    sampling_rate = float(sampling_rate)
                except (ValueError, TypeError):
                    sampling_rate = 1.0
                
                time_data = np.arange(len(data)) / sampling_rate
            
        return data, time_data, metadata
    
    def retrieve_multiple_channels(self, 
                                  diag_name: str,
                                  shot: int,
                                  subshot: int,
                                  channels: List[int],
                                  time_axis: bool = True) -> Dict[int, LHDData]:
        """
        Retrieve data for multiple channels from the same shot.
        Time axis is shared across all channels for efficiency.
        
        Args:
            diag_name: Diagnostic name
            shot: Shot number
            subshot: Sub-shot number
            channels: List of channel numbers
            time_axis: Generate time axis information
            
        Returns:
            Dictionary mapping channel numbers to LHDData objects
        """
        results = {}
        shared_time = None
        shared_metadata = {}
        
        for i, channel in enumerate(channels):
            try:
                # Prepare options
                options = []
                if time_axis:
                    options.append('-T')
                
                # Generate unique file prefix
                file_prefix = f"retrieve_{diag_name}_{shot}_{subshot}_{channel}"
                
                try:
                    dat_file, prm_file, time_file = self._run_retrieve(
                        diag_name=diag_name,
                        shot_no=shot,
                        subshot_no=subshot,
                        ch_no_name=channel,
                        file_prefix=file_prefix,
                        options=options
                    )
                    
                    # Parse the output files
                    data, time_data, metadata = self._parse_retrieve_files(dat_file, prm_file, time_file, None)
                    
                    # Use shared time for all channels (time should be identical across channels)
                    if i == 0:
                        # First channel: store time and metadata as shared
                        shared_time = time_data
                        shared_metadata = {k: v for k, v in metadata.items() if k not in ['channel']}
                    
                    results[channel] = LHDData(
                        data=data,
                        time=shared_time,  # Use shared time for all channels
                        metadata={
                            'diag_name': diag_name,
                            'shot': shot,
                            'subshot': subshot,
                            'channel': channel,
                            'time_axis': time_axis,
                            **shared_metadata  # Use shared metadata (excluding channel-specific data)
                        },
                        description=f"{diag_name} Shot {shot}.{subshot}, Channel {channel}"
                    )
                    
                finally:
                    # Clean up temporary files for this channel
                    self._cleanup_temporary_files(file_prefix)
                    
            except Exception as e:
                warnings.warn(f"Failed to retrieve channel {channel}: {e}")
                continue
        
        return results
    