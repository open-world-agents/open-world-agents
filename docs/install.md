# Installation Guide

## Quick Start (Recommended)

For most users who want to use Open World Agents without modifying the source code, installation is straightforward:

### Option 1: Full Installation with Video Processing

If you need **desktop recording, screen capture, or video processing capabilities**, use conda:

```bash
conda install owa
```

This installs the complete `owa` meta-package with all dependencies including GStreamer for high-performance video processing.

### Option 2: Headless Installation

For **data processing, ML training, or headless servers** without video capture needs:

```bash
pip install owa
```

This installs all core functionality except video processing components.

!!! tip "When to use conda vs pip"
    
    - **Use `conda install owa`** if you need:
        - Desktop recording with `ocap`
        - Real-time screen capture
        - Video processing capabilities
        - Complete out-of-the-box experience
    
    - **Use `pip install owa`** if you:
        - Only need data processing/analysis
        - Are on a headless server
        - Don't require video capture functionality

## Available Packages

All OWA packages follow lockstep versioning and are available on both PyPI and conda-forge:

| Name | PyPI | Conda | Description |
|------|------|-------|-------------|
| [`owa`](pyproject.toml) | [![owa](https://img.shields.io/pypi/v/owa?label=owa)](https://pypi.org/project/owa/) | [![owa](https://img.shields.io/conda/vn/conda-forge/owa?label=conda)](https://anaconda.org/conda-forge/owa) | **Meta-package** with all core components |
| [`owa-core`](projects/owa-core) | [![owa-core](https://img.shields.io/pypi/v/owa-core?label=owa-core)](https://pypi.org/project/owa-core/) | [![owa-core](https://img.shields.io/conda/vn/conda-forge/owa-core?label=conda)](https://anaconda.org/conda-forge/owa-core) | Framework foundation with registry system |
| [`owa-cli`](projects/owa-cli) | [![owa-cli](https://img.shields.io/pypi/v/owa-cli?label=owa-cli)](https://pypi.org/project/owa-cli/) | [![owa-cli](https://img.shields.io/conda/vn/conda-forge/owa-cli?label=conda)](https://anaconda.org/conda-forge/owa-cli) | Command-line tools (`owl`) for data analysis |
| [`mcap-owa-support`](projects/mcap-owa-support) | [![mcap-owa-support](https://img.shields.io/pypi/v/mcap-owa-support?label=mcap-owa-support)](https://pypi.org/project/mcap-owa-support/) | [![mcap-owa-support](https://img.shields.io/conda/vn/conda-forge/mcap-owa-support?label=conda)](https://anaconda.org/conda-forge/mcap-owa-support) | OWAMcap format support and utilities |
| [`ocap`](projects/ocap) 🎥 | [![ocap](https://img.shields.io/pypi/v/ocap?label=ocap)](https://pypi.org/project/ocap/) | [![ocap](https://img.shields.io/conda/vn/conda-forge/ocap?label=conda)](https://anaconda.org/conda-forge/ocap) | Desktop recorder for multimodal data capture |
| [`owa-env-desktop`](projects/owa-env-desktop) | [![owa-env-desktop](https://img.shields.io/pypi/v/owa-env-desktop?label=owa-env-desktop)](https://pypi.org/project/owa-env-desktop/) | [![owa-env-desktop](https://img.shields.io/conda/vn/conda-forge/owa-env-desktop?label=conda)](https://anaconda.org/conda-forge/owa-env-desktop) | Mouse, keyboard, window event handling |
| [`owa-env-gst`](projects/owa-env-gst) 🎥 | [![owa-env-gst](https://img.shields.io/pypi/v/owa-env-gst?label=owa-env-gst)](https://pypi.org/project/owa-env-gst/) | [![owa-env-gst](https://img.shields.io/conda/vn/conda-forge/owa-env-gst?label=conda)](https://anaconda.org/conda-forge/owa-env-gst) | GStreamer-powered screen capture (**6x faster**) |

> 🎥 **Video Processing Packages**: Packages marked with 🎥 require GStreamer for full functionality. Use `conda install` for complete features, `pip install` works for basic functionality.

## Development Installation (Editable)

!!! info "For Contributors and Developers"
    
    This section is for users who want to modify the source code, contribute to the project, or need the latest development features.

### Prerequisites

Before proceeding with development installation, ensure you have the necessary tools:

1. **Git**: For cloning the repository
2. **Python 3.11+**: Required for all OWA packages
3. **Virtual Environment Tool**: We recommend conda/mamba for complete functionality

### Step 1: Setup Virtual Environment

=== "conda/mamba (Recommended)"

    1. Install miniforge following the [installation guide](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install):
        ```sh
        # Download and install miniforge
        # This provides both conda and mamba (faster conda)
        ```

    2. Create and activate your environment:
        ```sh
        conda create -n owa-dev python=3.11 -y
        conda activate owa-dev
        ```

    3. **(Required for video processing)** Install GStreamer dependencies:
        ```sh
        # Clone the repo first to access environment.yml
        git clone https://github.com/open-world-agents/open-world-agents
        cd open-world-agents
        
        # Install GStreamer and related dependencies
        mamba env update --name owa-dev --file projects/owa-env-gst/environment.yml
        ```

=== "Other Virtual Environments"

    You can use other virtual environment tools (venv, virtualenv, poetry, etc.), but:
    
    - **GStreamer must be installed separately** for video processing functionality
    - **Installation complexity increases** due to GStreamer's native dependencies
    - **We recommend conda/mamba** for the best development experience

### Step 2: Clone and Setup Development Tools

```sh
# Clone the repository
git clone https://github.com/open-world-agents/open-world-agents
cd open-world-agents

# Install uv (fast Python package manager)
pip install uv

# Install virtual-uv for easier monorepo management
pip install virtual-uv
```

### Step 3: Install in Editable Mode

=== "uv + virtual-uv (Recommended)"

    ```sh
    # Ensure you're in the project root and environment is activated
    cd open-world-agents
    conda activate owa-dev  # or your environment name
    
    # Install all packages in editable mode
    vuv install
    ```

    !!! tip
        `vuv` (virtual-uv) handles the complex dependency resolution for our monorepo structure and installs all packages in the correct order.

=== "uv (Simple)"

    ```sh
    # Install with inexact dependency resolution
    uv sync --inexact
    ```

=== "pip (Manual)"

    ```sh
    # Install in correct order (dependency order matters with pip)
    pip install -e projects/owa-core
    pip install -e projects/mcap-owa-support
    pip install -e projects/owa-env-desktop
    pip install -e projects/owa-env-gst  # Requires GStreamer
    pip install -e projects/owa-cli
    pip install -e projects/ocap
    ```

    !!! warning "Installation Order Matters"
        When using `pip` instead of `uv`, the installation order is critical because `pip` cannot resolve the monorepo dependencies specified in `[tool.uv.sources]`.

### Step 4: Verify Installation

```sh
# Test core functionality
python -c "from owa.core.registry import CALLABLES; print('✅ Core installed')"

# Test CLI tools
owl --help
ocap --help

# Test GStreamer install if you need it
python -c "import gi; gi.require_version('Gst', '1.0'); print('✅ GStreamer OK')"
```

## Troubleshooting

### GStreamer Issues

If you encounter GStreamer-related errors:

1. **Ensure conda environment**: GStreamer installation works best through conda
2. **Update environment file**:
   ```sh
   mamba env update --name your-env --file projects/owa-env-gst/environment.yml
   ```
3. **Check GStreamer installation**:
   ```sh
   python -c "import gi; gi.require_version('Gst', '1.0'); print('✅ GStreamer OK')"
   ```

### Virtual Environment Issues

- **Always activate your environment** before running `vuv` or installation commands
- **Use absolute paths** if you encounter import issues
- **Reinstall virtual-uv** if you encounter dependency resolution problems:
  ```sh
  pip uninstall virtual-uv
  pip install virtual-uv
  ```

### Package Version Conflicts

OWA uses lockstep versioning. If you encounter version conflicts:

```sh
# Check installed versions
pip list | grep owa

# Reinstall with matching versions
pip install owa-core==0.3.2 owa-cli==0.3.2 owa-env-desktop==0.3.2
```

## What's Next?

After installation, check out:

- **[Quick Start Guide](../quickstart.md)**: Basic usage examples
- **[Environment Guide](../env/)**: Understanding the plugin system
- **[Data Format Guide](../data/)**: Working with OWAMcap files
- **[Contributing Guide](../contributing.md)**: How to contribute to the project