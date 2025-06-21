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

Open World Agents is a comprehensive framework for building AI agents that interact with desktop applications through vision, keyboard, and mouse control. Complete toolkit from data capture to model training and evaluation:

- **OWA Core & Environment**: Asynchronous, event-driven interface for real-time agents with dynamic plugin activation
- **Data Capture & Format**: High-performance desktop recording with `OWAMcap` format - a specialized file format that captures screen recordings, keyboard/mouse events, and window information with nanosecond precision, powered by [mcap](https://mcap.dev/)
- **Environment Plugins**: Pre-built plugins for desktop automation, screen capture, and more
- **CLI Tools**: Command-line utilities for recording, analyzing, and managing agent data

## What Can You Build?

**Anything that runs on desktop.** If a human can do it on a computer, you can build an AI agent to automate it.

🤖 **Desktop Automation**: Navigate applications, automate workflows, interact with any software  
🎮 **Game AI**: Master complex games through visual understanding and real-time decision making  
📊 **Training Datasets**: Capture high-quality human-computer interaction data for foundation models  
🤗 **Community Datasets**: Access and contribute to growing OWAMcap datasets on HuggingFace  
📈 **Benchmarks**: Create and evaluate desktop agent performance across diverse tasks  

## Project Structure

The repository is organized as a monorepo with multiple sub-repositories under the `projects/` directory. Each sub-repository is a self-contained Python package installable via `pip` or [`uv`](https://docs.astral.sh/uv/) and follows namespace packaging conventions.

```
open-world-agents/
├── projects/
│   ├── mcap-owa-support/     # OWAMcap format support
│   ├── owa-core/             # Core framework and registry system
│   ├── owa-msgs/             # Core message definitions with automatic discovery
│   ├── owa-cli/              # Command-line tools (ocap, owl)
│   ├── owa-env-desktop/      # Desktop environment plugin
│   ├── owa-env-example/      # Example environment implementations
│   ├── owa-env-gst/          # GStreamer-based screen capture
│   └── [your-plugin]/        # Contribute your own plugins!
├── docs/                     # Documentation
└── README.md
```

## Core Packages

[![owa](https://img.shields.io/pypi/v/owa?label=owa)](https://pypi.org/project/owa/) [![owa](https://img.shields.io/conda/vn/conda-forge/owa?label=conda)](https://anaconda.org/conda-forge/owa)

The easiest way to get started is to install the [**owa**](pyproject.toml) meta-package, which includes all core components and environment plugins:

```bash
pip install owa
```

All OWA packages use namespace packaging and are installed in the `owa` namespace (e.g., `owa.core`, `owa.cli`, `owa.env.desktop`). For more detail, see [Packaging namespace packages](https://packaging.python.org/en/latest/guides/packaging-namespace-packages/). We recommend using [`uv`](https://docs.astral.sh/uv/) as the package manager.


| Name | Release in PyPI | Conda | Description |
|------|-----------------|-------|-------------|
| [`owa.core`](projects/owa-core) | [![owa-core](https://img.shields.io/pypi/v/owa-core?label=owa-core)](https://pypi.org/project/owa-core/) | [![owa-core](https://img.shields.io/conda/vn/conda-forge/owa-core?label=conda)](https://anaconda.org/conda-forge/owa-core) | Framework foundation with registry system |
| [`owa.msgs`](projects/owa-msgs) | [![owa-msgs](https://img.shields.io/pypi/v/owa-msgs?label=owa-msgs)](https://pypi.org/project/owa-msgs/) | [![owa-msgs](https://img.shields.io/conda/vn/conda-forge/owa-msgs?label=conda)](https://anaconda.org/conda-forge/owa-msgs) | Core message definitions with automatic discovery |
| [`owa.cli`](projects/owa-cli) | [![owa-cli](https://img.shields.io/pypi/v/owa-cli?label=owa-cli)](https://pypi.org/project/owa-cli/) | [![owa-cli](https://img.shields.io/conda/vn/conda-forge/owa-cli?label=conda)](https://anaconda.org/conda-forge/owa-cli) | Command-line tools (`owl`) for data analysis |
| [`mcap-owa-support`](projects/mcap-owa-support) | [![mcap-owa-support](https://img.shields.io/pypi/v/mcap-owa-support?label=mcap-owa-support)](https://pypi.org/project/mcap-owa-support/) | [![mcap-owa-support](https://img.shields.io/conda/vn/conda-forge/mcap-owa-support?label=conda)](https://anaconda.org/conda-forge/mcap-owa-support) | OWAMcap format support and utilities |
| [`ocap`](projects/ocap) 🎥 | [![ocap](https://img.shields.io/pypi/v/ocap?label=ocap)](https://pypi.org/project/ocap/) | [![ocap](https://img.shields.io/conda/vn/conda-forge/ocap?label=conda)](https://anaconda.org/conda-forge/ocap) | Desktop recorder for multimodal data capture |
| [`owa.env.desktop`](projects/owa-env-desktop) | [![owa-env-desktop](https://img.shields.io/pypi/v/owa-env-desktop?label=owa-env-desktop)](https://pypi.org/project/owa-env-desktop/) | [![owa-env-desktop](https://img.shields.io/conda/vn/conda-forge/owa-env-desktop?label=conda)](https://anaconda.org/conda-forge/owa-env-desktop) | Mouse, keyboard, window event handling |
| [`owa.env.gst`](projects/owa-env-gst) 🎥 | [![owa-env-gst](https://img.shields.io/pypi/v/owa-env-gst?label=owa-env-gst)](https://pypi.org/project/owa-env-gst/) | [![owa-env-gst](https://img.shields.io/conda/vn/conda-forge/owa-env-gst?label=conda)](https://anaconda.org/conda-forge/owa-env-gst) | GStreamer-powered screen capture (**[6x faster](#high-performance-screen-capture)**) |
| [`owa.env.example`](projects/owa-env-example) | - | - | Reference implementations for learning |

> 🎥 **Video Processing Packages**: Packages marked with 🎥 require GStreamer dependencies. Install `conda install open-world-agents::gstreamer-bundle` first for full functionality.

> 📦 **Lockstep Versioning**: All first-party OWA packages follow lockstep versioning, meaning they share the same version number to ensure compatibility and simplify dependency management.

> 💡 **Extensible Design**: Built for the community! Easily create custom plugins like `owa-env-minecraft` or `owa-env-web` to extend functionality.

## Community Packages

**Help us grow the ecosystem!** 🌱 Community-contributed environment plugins extend OWA's capabilities to specialized domains.

*Example plugin ideas from the community:*

| Example Name | Description | 
|--------------|-------------|
| `owa.env.minecraft` | Minecraft automation & bot framework |
| `owa.env.web` | Browser automation via WebDriver |
| `owa.env.mobile` | Android/iOS device control |
| `owa.env.cad` | CAD software automation (AutoCAD, SolidWorks) |
| `owa.env.trading` | Financial trading platform integration |

> 💡 **Want to contribute?** Check our [Plugin Development Guide](docs/env/custom_plugins.md) to create your own `owa.env.*` package!
> 
> 💭 **These are just examples!** The community decides what plugins to build. Propose your own ideas or create plugins for any domain you're passionate about.

### Desktop Recording with `ocap`

**ocap** (Omnimodal CAPture) is a high-performance desktop recorder that captures screen video, audio, keyboard/mouse events, and window events in synchronized formats. Built with Windows APIs and GStreamer for hardware-accelerated recording with H265/HEVC encoding.

- **Complete recording**: Video + audio + keyboard/mouse + window events
- **High performance**: Hardware-accelerated, ~100MB/min for 1080p
- **Simple usage**: `ocap my-recording` (stop with Ctrl+C)
- **Modern formats**: MKV for video, MCAP for events

> 📖 **Detailed Documentation**: See [Desktop Recording Guide](docs/data/ocap.md) for complete setup, usage examples, and troubleshooting.

## Quick Start

### Basic Environment Usage

```python
import time
from owa.core import CALLABLES, LISTENERS, MESSAGES

# Components and messages automatically available - no activation needed!

def callback():
    time_ns = CALLABLES["std/time_ns"]()
    print(f"Current time: {time_ns}")

# Access message types through the global registry
KeyboardEvent = MESSAGES['desktop/KeyboardEvent']
print(f"Available message: {KeyboardEvent}")

# Create a listener for std/tick event (every 1 second)
tick = LISTENERS["std/tick"]().configure(callback=callback, interval=1)

# Start listening
tick.start()
time.sleep(2)
tick.stop(), tick.join()
```

### High-Performance Screen Capture

```python
import time
from owa.core import CALLABLES, LISTENERS, MESSAGES

# Components and messages automatically available - no activation needed!

def on_screen_update(frame, metrics):
    print(f"📸 New frame: {frame.frame_arr.shape}")
    print(f"⚡ Latency: {metrics.latency*1000:.1f}ms")

    # Access screen message type from registry
    ScreenCaptured = MESSAGES['desktop/ScreenCaptured']
    print(f"Frame message type: {ScreenCaptured}")

# Start real-time screen capture
screen = LISTENERS["gst/screen"]().configure(
    callback=on_screen_update, fps=60, show_cursor=True
)

with screen.session:
    print("🎯 Agent is watching your screen...")
    time.sleep(5)
```

### Plugin Management with CLI

Explore and manage plugins using the enhanced `owl env` command:

```bash
# List all discovered plugins with enhanced display
$ owl env list --details --table

# Show detailed plugin information with component inspection
$ owl env show example --components --inspect add

# Search for components across all plugins
$ owl env search "mouse.*click" --table

# Quick exploration shortcuts
$ owl env ls desktop                              # Quick namespace exploration
$ owl env find keyboard                           # Quick component search
$ owl env namespaces                              # List all available namespaces

# Ecosystem analysis and health monitoring
$ owl env stats                                   # Show ecosystem statistics
$ owl env health                                  # Perform health check
```

### Message Management with CLI

Explore and manage message types using the new `owl messages` command:

```bash
# List all available message types
$ owl messages list

# Show detailed message schema
$ owl messages show desktop/KeyboardEvent

# Search for specific message types
$ owl messages search keyboard

# Validate message definitions
$ owl messages validate
```

Powered by the powerful Gstreamer and Windows API, our implementation is **6x** faster than comparatives.

| **Library**        | **Avg. Time per Frame** | **Relative Speed**    |
|--------------------|------------------------|-----------------------|
| **owa.env.gst**   | **5.7 ms**              | ⚡ **1× (Fastest)**    |
| `pyscreenshot`    | 33 ms                   | 🚶‍♂️ 5.8× slower       |
| `PIL`             | 34 ms                   | 🚶‍♂️ 6.0× slower       |
| `MSS`             | 37 ms                   | 🚶‍♂️ 6.5× slower       |
| `PyQt5`           | 137 ms                  | 🐢 24× slower         |

📌 **Tested on:** Intel i5-11400, GTX 1650  

Not only does `owa.env.gst` **achieve higher FPS**, but it also maintains **lower CPU/GPU usage**, making it the ideal choice for screen recording. Same applies for `ocap`, since it internally imports `owa.env.gst`.

### Desktop Recording & Dataset Sharing

Record your desktop usage data and share with the community:

```bash
# Install GStreamer dependencies (for video recording) and ocap
conda install open-world-agents::gstreamer-bundle && pip install ocap

# Record desktop activity (includes video, audio, events)
ocap my-session

# Upload to HuggingFace, browse community datasets!
# Visit: https://huggingface.co/datasets?other=OWA
```

### Access Community Datasets

> 🚧 **TODO**: Community dataset access functionality is under development.

```python
# Load datasets from HuggingFace
from owa.data import load_dataset

# Browse available OWAMcap datasets
datasets = load_dataset.list_available(format="OWA")

# Load a specific dataset
data = load_dataset("open-world-agents/example_dataset")
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
        (1) keyboard/state    9 msgs (1.02 Hz)    : desktop/KeyboardState [jsonschema]
        (2) mouse/state       9 msgs (1.02 Hz)    : desktop/MouseState [jsonschema]
        (3) window            9 msgs (1.02 Hz)    : desktop/WindowInfo [jsonschema]
        (4) screen          523 msgs (59.35 Hz)   : desktop/ScreenCaptured [jsonschema]
        (5) mouse           510 msgs (57.87 Hz)   : desktop/MouseEvent [jsonschema]
        (6) keyboard          2 msgs (0.23 Hz)    : desktop/KeyboardEvent [jsonschema]
channels: 6
attachments: 0
metadata: 0
```

## Installation

### Quick Start

```bash
# Install all OWA packages
pip install owa

# For video recording/processing, install GStreamer dependencies first:
conda install open-world-agents::gstreamer-bundle
pip install owa
```

> 💡 **When do you need GStreamer?**
> - **Video recording** with `ocap` desktop recorder
> - **Real-time screen capture** with `owa.env.gst`
> - **Video processing** capabilities
>
> **Skip GStreamer if you only need:**
> - Data processing and analysis
> - ML training on existing datasets
> - Headless server environments

### Editable Install (Development)

For development or contributing to the project, you can install packages in editable mode. For detailed development setup instructions, see the [Installation Guide](docs/install.md).


## Features

- **🔄 Asynchronous Processing**: Real-time event handling with Callables, Listeners, and Runnables
- **🧩 Zero-Configuration Plugin System**: Automatic plugin discovery via Entry Points
- **📊 High-Performance Data**: 6x faster screen capture with GStreamer integration
- **🤗 HuggingFace Ecosystem**: Access growing collection of community OWAMcap datasets
- **🗂️ OWAMcap Format**: Specialized file format capturing complete desktop interactions (screen + keyboard + mouse + windows) with perfect synchronization
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