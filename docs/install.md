# Installation Guide

## Quick Start (Recommended)

For most users who want to use Open World Agents, installation is straightforward:

### Option 1: Full Installation with Video Processing

If you need **desktop recording, screen capture, or video processing capabilities**:

```bash
# Install GStreamer dependencies first
conda install open-world-agents::gstreamer-bundle

# Then install all OWA packages
pip install owa
```

### Option 2: Headless Installation

For **data processing, ML training, or headless servers** without video capture needs:

```bash
pip install owa
```

!!! tip "When to use GStreamer"
    
    **Install GStreamer if you need:**

    - Desktop recording with `ocap`
    - Real-time screen capture with `owa.env.gst`
    - Video processing capabilities
    - Complete multimodal data capture
    
    **Skip GStreamer if you only need:**

    - Data processing and analysis
    - ML training on existing datasets
    - Headless server environments
    - Working with pre-recorded MCAP files

## Available Packages

All OWA packages are pure Python and available on PyPI. Install via `pip install owa` for all components:

| Name | PyPI | Description |
|------|------|-------------|
| [`owa`](https://github.com/open-world-agents/open-world-agents/blob/main/pyproject.toml) | [![owa](https://img.shields.io/pypi/v/owa?label=owa)](https://pypi.org/project/owa/) | **Meta-package** with all core components |
| [`owa-core`](https://github.com/open-world-agents/open-world-agents/tree/main/projects/owa-core) | [![owa-core](https://img.shields.io/pypi/v/owa-core?label=owa-core)](https://pypi.org/project/owa-core/) | Framework foundation with registry system |
| [`owa-cli`](https://github.com/open-world-agents/open-world-agents/tree/main/projects/owa-cli) | [![owa-cli](https://img.shields.io/pypi/v/owa-cli?label=owa-cli)](https://pypi.org/project/owa-cli/) | Command-line tools (`owl`) for data analysis |
| [`mcap-owa-support`](https://github.com/open-world-agents/open-world-agents/tree/main/projects/mcap-owa-support) | [![mcap-owa-support](https://img.shields.io/pypi/v/mcap-owa-support?label=mcap-owa-support)](https://pypi.org/project/mcap-owa-support/) | OWAMcap format support and utilities |
| [`ocap`](https://github.com/open-world-agents/open-world-agents/tree/main/projects/ocap) 🎥 | [![ocap](https://img.shields.io/pypi/v/ocap?label=ocap)](https://pypi.org/project/ocap/) | Desktop recorder for multimodal data capture |
| [`owa-env-desktop`](https://github.com/open-world-agents/open-world-agents/tree/main/projects/owa-env-desktop) | [![owa-env-desktop](https://img.shields.io/pypi/v/owa-env-desktop?label=owa-env-desktop)](https://pypi.org/project/owa-env-desktop/) | Mouse, keyboard, window event handling |
| [`owa-env-gst`](https://github.com/open-world-agents/open-world-agents/tree/main/projects/owa-env-gst) 🎥 | [![owa-env-gst](https://img.shields.io/pypi/v/owa-env-gst?label=owa-env-gst)](https://pypi.org/project/owa-env-gst/) | GStreamer-powered screen capture (**6x faster**) |

> 🎥 **Video Processing Packages**: Packages marked with 🎥 require GStreamer dependencies. Install `conda install open-world-agents::gstreamer-bundle` first for full functionality.

### GStreamer Bundle

For video processing capabilities, install the GStreamer bundle separately:

```bash
conda install open-world-agents::gstreamer-bundle
```

This bundle includes all necessary GStreamer dependencies (pygobject, gst-python, gst-plugins, etc.) that are complex to install via pip.

## Development Installation (Editable)

!!! info "For Contributors and Developers"
    
    This section is for users who want to modify the source code, contribute to the project, or need the latest development features.

### Prerequisites

Before proceeding with development installation, ensure you have the necessary tools:

1. **Git**: For cloning the repository
2. **Python 3.11+**: Required for all OWA packages
3. **Virtual Environment Tool**: We recommend conda/mamba for GStreamer support

### Step 1: Setup Virtual Environment

=== "conda/mamba (Recommended)"

    1. Install miniforge following the [installation guide](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install):
        ```sh
        # Download and install miniforge
        # This provides both conda and mamba (faster conda)
        ```

    2. Create and activate your environment:
        ```sh
        conda create -n owa python=3.11 -y
        conda activate owa
        ```

    3. **(Optional for video processing)** Install GStreamer dependencies:
        ```sh
        # Install GStreamer bundle for video processing
        conda install open-world-agents::gstreamer-bundle
        ```

=== "Other Virtual Environments"

    You can use other virtual environment tools (venv, virtualenv, poetry, etc.), but:
    
    - **GStreamer must be installed separately** for video processing functionality, which is not easy without `conda`
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
    conda activate owa  # or your environment name
    
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
owl env list  # List discovered plugins
ocap --help

# Test GStreamer (if installed)
python -c "import gi; gi.require_version('Gst', '1.0'); print('✅ GStreamer OK')"
```

## Troubleshooting

### GStreamer Issues

If you encounter GStreamer-related errors:

1. **Install GStreamer bundle**:
   ```sh
   conda install open-world-agents::gstreamer-bundle
   ```
2. **Verify GStreamer installation**:
   ```sh
   python -c "import gi; gi.require_version('Gst', '1.0'); print('✅ GStreamer OK')"
   ```
3. **Restart your Python environment** after installing GStreamer dependencies

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

# Reinstall all packages with matching versions
pip install --upgrade owa
```

### Import Errors

If you encounter import errors after installation:

1. **Ensure Python environment is activated**
2. **Restart Python kernel/terminal** after installing packages
3. **Verify installation**: `pip list | grep owa`