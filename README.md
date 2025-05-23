<div align="center">
  <img src="docs/images/owa-logo.jpg" alt="Open World Agents" width="300"/>
  
  # 🚀 Open World Agents
  
  **Everything you need to build state-of-the-art foundation multimodal desktop agent, end-to-end.**
  
  [![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://open-world-agents.github.io/open-world-agents/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
  [![GitHub stars](https://img.shields.io/github/stars/open-world-agents/open-world-agents?style=social)](https://github.com/open-world-agents/open-world-agents/stargazers)
  
</div>

## Overview

**Everything you need to build state-of-the-art foundation multimodal desktop agents, end-to-end.**

Open World Agents is a comprehensive framework that streamlines your agent's lifecycle from data capture to model training and real-time evaluation. The repository contains essential components for building multimodal desktop agents:

- **OWA Core & Environment**: Asynchronous, event-driven environmental interface for real-time agents with dynamic plugin activation
- **Data Capture & Format**: High-performance desktop recording with the `OWAMcap` format powered by [mcap](https://mcap.dev/)
- **Environment Plugins**: Pre-built plugins for desktop automation, screen capture, and more
- **CLI Tools**: Command-line utilities for recording, analyzing, and managing agent data

## What Can You Build?

🤖 **Desktop Automation Agents**: Build AI agents that can navigate complex desktop applications, automate workflows, and interact with any software

📊 **Training Datasets**: Capture high-quality multimodal datasets of human computer interactions for training foundation models

🤗 **Community Datasets**: Access a growing collection of open-source OWAMcap datasets on HuggingFace for immediate use

🎮 **Game AI**: Create agents that can play and master complex games by understanding visual interfaces and game mechanics

📈 **Benchmarks & Evaluation**: Create and run real-time benchmarks for desktop agents.

## Project Structure

The repository is organized as a monorepo with multiple sub-repositories under the `projects/` directory. Each sub-repository is a self-contained Python package installable via `pip` or [`uv`](https://docs.astral.sh/uv/) and follows namespace packaging conventions.

```
open-world-agents/
├── projects/
│   ├── mcap-owa-support/     # OWAMcap format support
│   ├── owa-core/             # Core framework and registry system
│   ├── owa-cli/              # Command-line tools (ocap, owl)
│   ├── owa-env-desktop/      # Desktop environment plugin
│   ├── owa-env-example/      # Example environment implementations
│   ├── owa-env-gst/          # GStreamer-based screen capture
│   └── [your-plugin]/        # Contribute your own plugins!
├── docs/                     # Documentation
└── README.md
```

## Python Packages

All OWA packages are installed in the `owa` namespace (e.g., `owa.core`, `owa.cli`, `owa.env.desktop`). We recommend using [`uv`](https://docs.astral.sh/uv/) as the package manager.

> 📦 **Lockstep Versioning**: All first-party OWA packages follow lockstep versioning, meaning they share the same version number to ensure compatibility and simplify dependency management.

### The `owa` meta-package

[![owa](https://img.shields.io/pypi/v/owa?label=owa)](https://pypi.org/project/owa/) [![owa](https://img.shields.io/conda/vn/conda-forge/owa?label=conda)](https://anaconda.org/conda-forge/owa)

The easiest way to get started is to install the [**owa**](pyproject.toml) meta-package, which includes all core components and environment plugins:

```bash
pip install owa
# or
conda install owa
```

This installs: `mcap-owa-support`, `ocap`, `owa-cli`, `owa-core`, `owa-env-desktop`, and `owa-env-gst`.

### Core Packages

| Name | Release in PyPI | Conda | Description |
|------|-----------------|-------|-------------|
| [`owa.core`](projects/owa-core) | [![owa-core](https://img.shields.io/pypi/v/owa-core?label=owa-core)](https://pypi.org/project/owa-core/) | [![owa-core](https://img.shields.io/conda/vn/conda-forge/owa-core?label=conda)](https://anaconda.org/conda-forge/owa-core) | Framework foundation with registry system |
| [`owa.cli`](projects/owa-cli) | [![owa-cli](https://img.shields.io/pypi/v/owa-cli?label=owa-cli)](https://pypi.org/project/owa-cli/) | [![owa-cli](https://img.shields.io/conda/vn/conda-forge/owa-cli?label=conda)](https://anaconda.org/conda-forge/owa-cli) | Command-line tools (`owl`) for data analysis |
| [`mcap-owa-support`](projects/mcap-owa-support) | [![mcap-owa-support](https://img.shields.io/pypi/v/mcap-owa-support?label=mcap-owa-support)](https://pypi.org/project/mcap-owa-support/) | [![mcap-owa-support](https://img.shields.io/conda/vn/conda-forge/mcap-owa-support?label=conda)](https://anaconda.org/conda-forge/mcap-owa-support) | OWAMcap format support and utilities |

### CLI Tools

| Name | Release in PyPI | Conda | Description |
|------|-----------------|-------|-------------|
| [`ocap`](projects/ocap) | [![ocap](https://img.shields.io/pypi/v/ocap?label=ocap)](https://pypi.org/project/ocap/) | [![ocap](https://img.shields.io/conda/vn/conda-forge/ocap?label=conda)](https://anaconda.org/conda-forge/ocap) | Desktop recorder for multimodal data capture |

> ⚠️ **GStreamer Required**: `ocap` requires GStreamer for video processing. Use `conda install owa-env-gst` for easy setup.

**ocap** (Omnimodal CAPture) is a high-performance desktop recorder that captures screen video, audio, keyboard/mouse events, and window events in synchronized formats. Built with Windows APIs and GStreamer for hardware-accelerated recording with H265/HEVC encoding.

- **Complete recording**: Video + audio + keyboard/mouse + window events
- **High performance**: Hardware-accelerated, ~100MB/min for 1080p
- **Simple usage**: `ocap my-recording` (stop with Ctrl+C)
- **Modern formats**: MKV for video, MCAP for events

### Environment Plugins

| Name | Release in PyPI | Conda | Description |
|------|-----------------|-------|-------------|
| [`owa.env.desktop`](projects/owa-env-desktop) | [![owa-env-desktop](https://img.shields.io/pypi/v/owa-env-desktop?label=owa-env-desktop)](https://pypi.org/project/owa-env-desktop/) | [![owa-env-desktop](https://img.shields.io/conda/vn/conda-forge/owa-env-desktop?label=conda)](https://anaconda.org/conda-forge/owa-env-desktop) | Mouse, keyboard, window event handling |
| [`owa.env.gst`](projects/owa-env-gst) | [![owa-env-gst](https://img.shields.io/pypi/v/owa-env-gst?label=owa-env-gst)](https://pypi.org/project/owa-env-gst/) | [![owa-env-gst](https://img.shields.io/conda/vn/conda-forge/owa-env-gst?label=conda)](https://anaconda.org/conda-forge/owa-env-gst) | GStreamer-powered screen capture (**6x faster**) |
| [`owa.env.example`](projects/owa-env-example) | - | - | Reference implementations for learning |

> ⚠️ **GStreamer Required**: Packages marked with video capabilities need GStreamer installed. To utilize full features, install with `conda`, not `pip`.

> 💡 **Extensible Design**: Built for the community! Easily create custom plugins like `owa-env-minecraft` or `owa-env-web` to extend functionality.

## Quick Start

### Basic Environment Usage

```python
import time
from owa.core.registry import CALLABLES, LISTENERS, activate_module

# Activate the standard environment module
activate_module("owa.env.std")

def callback():
    time_ns = CALLABLES["clock.time_ns"]()
    print(f"Current time: {time_ns}")

# Create a listener for clock/tick event (every 1 second)
tick = LISTENERS["clock/tick"]().configure(callback=callback, interval=1)

# Start listening
tick.start()
time.sleep(2)
tick.stop(), tick.join()
```

### Desktop Recording & Dataset Sharing

Record your desktop usage data and share with the community:

```bash
# Install desktop recorder
conda install ocap

# Record desktop activity (includes video, audio, events)
ocap my-session

# Upload to HuggingFace (coming soon)
owl dataset upload my-session.mcap --name "my-workflow-demo"

# Browse community datasets
# Visit: https://huggingface.co/datasets?other=owamcap
```

### Access Community Datasets

```python
# Load datasets from HuggingFace
from owa.data import load_dataset

# Browse available OWAMcap datasets
datasets = load_dataset.list_available(format="owamcap")

# Load a specific dataset
data = load_dataset("username/desktop-workflow-v1")
```

### Data Format Preview

```bash
$ owl mcap info example.mcap
library:   mcap-owa-support 0.3.2; mcap 1.2.2
profile:   owa
messages:  1062
duration:  8.8121584s
start:     2025-05-23T20:04:01.7269392+09:00 (1747998241.726939200)
end:       2025-05-23T20:04:10.5390976+09:00 (1747998250.539097600)
compression:
        zstd: [1/1 chunks] [113.42 KiB/17.52 KiB (84.55%)] [1.99 KiB/sec]
channels:
        (1) keyboard/state    9 msgs (1.02 Hz)    : owa.env.desktop.msg.KeyboardState [jsonschema]
        (2) mouse/state       9 msgs (1.02 Hz)    : owa.env.desktop.msg.MouseState [jsonschema]
        (3) window            9 msgs (1.02 Hz)    : owa.env.desktop.msg.WindowInfo [jsonschema]
        (4) screen          523 msgs (59.35 Hz)   : owa.env.gst.msg.ScreenEmitted [jsonschema]
        (5) mouse           510 msgs (57.87 Hz)   : owa.env.desktop.msg.MouseEvent [jsonschema]
        (6) keyboard          2 msgs (0.23 Hz)    : owa.env.desktop.msg.KeyboardEvent [jsonschema]
channels: 6
attachments: 0
metadata: 0
```

## Installation

### Quick Start

```bash
# Full installation with video processing capabilities and gstreamer
conda install owa

# For headless servers (data processing/ML training only)
pip install owa
```

> 💡 **GStreamer Dependencies**: 
> - **Need video recording/processing?** Use `conda install owa` or `conda install owa-env-gst`
> - **Headless server/data processing only?** `pip install owa` is sufficient
> - **Why conda for GStreamer?** GStreamer has complex native dependencies (pygobject, gst-python, gst-plugins, etc.) that conda handles automatically

### Editable Install (Development)

For development or contributing to the project, you can install packages in editable mode. For detailed development setup instructions, see the [Installation Guide](docs/install.md).


## Features

- **🔄 Asynchronous Processing**: Real-time event handling with Callables, Listeners, and Runnables
- **🧩 Dynamic Plugin System**: Runtime plugin activation and registration
- **📊 High-Performance Data**: 6x faster screen capture with GStreamer integration
- **🤗 HuggingFace Ecosystem**: Access growing collection of community OWAMcap datasets
- **🗂️ OWAMcap Format**: Self-contained, flexible multimodal data containers
- **🛠️ Extensible**: Community-driven plugin ecosystem

## Documentation

- **Full Documentation**: https://open-world-agents.github.io/open-world-agents/
- **Environment Guide**: [docs/env/](docs/env/)
- **Data Format**: [docs/data/](docs/data/)
- **Plugin Development**: [docs/env/custom_plugins.md](docs/env/custom_plugins.md)

## Contributing

We welcome contributions! Whether you're:
- Building new environment plugins
- Improving performance
- Adding documentation
- Reporting bugs

Please see our [Contributing Guide](docs/contributing.md) for details.

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**🚧 Work in Progress**: We're actively developing this framework. Stay tuned for more updates and examples!