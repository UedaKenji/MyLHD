# mylhd - LHD Data Collection and Analysis Package

A comprehensive Python package for collecting and analyzing data from the Large Helical Device (LHD) plasma physics experiment.

## Features

- **anadata**: Core module for LHD kaiseki data format parsing and retrieval from open data servers
- **labcom_retrieve**: Package for retrieving measurement data from LABCOM systems via Retrieve.exe
- **cts_utls**: CTS (Collective Thomson Scattering) analysis tools and utilities

## Installation

### From Source

Clone the repository and install in development mode:

```bash
git clone https://github.com/UedaKenji/MyLHD.git
cd MyLHD
pip install -e .
```

### Dependencies

The package requires Python 3.7+ and the following dependencies:

- numpy >= 1.19.0
- pandas >= 1.2.0
- matplotlib >= 3.3.0
- scipy >= 1.6.0
- urllib3 >= 1.26.0

### Development Installation

For development with additional tools:

```bash
pip install -e ".[dev]"
```

This includes testing and code quality tools:
- pytest >= 6.0
- pytest-cov >= 2.0
- black >= 21.0
- flake8 >= 3.9
- mypy >= 0.910

## Environment Requirements

### Windows/WSL Environment

The `labcom_retrieve` package requires Windows or WSL environment to access Retrieve.exe:
- Default paths: `/mnt/c/LABCOM/Retrieve/bin/Retrieve.exe`, `/mnt/c/LHD/Retrieve/Retrieve.exe`
- Working directory automatically set to Retrieve.exe parent directory
- Timeout: 5 minutes per retrieval operation

## Quick Start

### Kaiseki Data Retrieval

```python
import mylhd

# Retrieve data from open server
data = mylhd.KaisekiData.retrieve_opendata('diag_name', shot_no=123456, subno=1)
```

### LABCOM Data Retrieval

```python
import mylhd

# Initialize retriever
retriever = mylhd.LHDRetriever()

# Retrieve single channel data
lhd_data = retriever.retrieve_data(diag='Mag', shotno=139400, subshot=1, channel=32, time_axis=True)

# Retrieve multiple channels
multi_data = retriever.retrieve_multiple_channels('Mag', 139400, 1, [1,2,3,4])
```

### CTS Analysis

```python
import mylhd.cts_utls as cts

# Generate CTS spectrogram plot
cts.save_plot(shot_num=139400, diag='CTS', channel=1)
```

## Command Line Tools

After installation, the following command line tool is available:

```bash
ctsviewer  # CTS oscilloscope data viewer
```

## License

MIT License

## Author

Kenji Ueda (kenji.ueda@nifs.ac.jp)
