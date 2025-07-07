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

- **OWA Core & Environment**: Asynchronous, event-driven interface for real-time agents with zero-configuration plugin system
- **Data Capture & Format**: High-performance desktop recording with `OWAMcap` format - a [universal standard](https://open-world-agents.github.io/open-world-agents/data/getting-started/why-owamcap/) with flexible MediaRef system for efficient external media references, powered by [mcap](https://mcap.dev/)
- **Environment Plugins**: Pre-built plugins for desktop automation, screen capture, and more
- **CLI Tools**: Command-line utilities for recording, analyzing, and managing agent data

## What Can You Build?

**Anything that runs on desktop.** If a human can do it on a computer, you can build an AI agent to automate it.

🤖 **Desktop Automation**: Navigate applications, automate workflows, interact with any software  
🎮 **Game AI**: Master complex games through visual understanding and real-time decision making  
📊 **Training Datasets**: Capture high-quality human-computer interaction data for foundation models  
🤗 **Community Datasets**: Access and contribute to growing [OWAMcap datasets](https://huggingface.co/datasets?other=OWA) on HuggingFace  
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

> 💡 **Want to contribute?** Check our [Plugin Development Guide](https://open-world-agents.github.io/open-world-agents/env/custom_plugins/) to create your own `owa.env.*` package!
> 
> 💭 **These are just examples!** The community decides what plugins to build. Propose your own ideas or create plugins for any domain you're passionate about.

### Desktop Recording with `ocap`

**ocap** (Omnimodal CAPture) is a high-performance desktop recorder that captures screen video, audio, keyboard/mouse events, and window events in synchronized formats. Built with Windows APIs and GStreamer for hardware-accelerated recording with H265/HEVC encoding.

- **Complete recording**: Video + audio + keyboard/mouse + window events
- **High performance**: Hardware-accelerated, ~100MB/min for 1080p
- **Simple usage**: `ocap my-recording` (stop with Ctrl+C)
- **Modern formats**: [OWAMcap](https://open-world-agents.github.io/open-world-agents/data/getting-started/why-owamcap/) with flexible MediaRef system (supports MKV, images, URLs, embedded data)

> 📖 **Detailed Documentation**: See [Desktop Recording Guide](https://open-world-agents.github.io/open-world-agents/data/getting-started/recording-data/) for complete setup, usage examples, and troubleshooting.

## Quick Start

### Environment Usage

```python
import time
from owa.core import CALLABLES, LISTENERS, MESSAGES

# Components automatically available - zero configuration!

# Time and desktop automation
current_time = CALLABLES["std/time_ns"]()
screen = CALLABLES["desktop/screen.capture"]()
CALLABLES["desktop/mouse.click"]("left", 2)  # Double-click

# Event monitoring with context manager
def on_tick():
    print(f"Tick: {CALLABLES['std/time_ns']()}")

with LISTENERS["std/tick"]().configure(callback=on_tick, interval=1).session:
    time.sleep(3)  # Prints every second for 3 seconds

# Message types
KeyboardEvent = MESSAGES["desktop/KeyboardEvent"]
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

📊 **[See detailed benchmarks and methodology →](https://open-world-agents.github.io/open-world-agents/env/plugins/gst#performance)**

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

Browse and load community-contributed OWAMcap datasets directly from HuggingFace:

```python
# Load datasets from HuggingFace
from owa.data import load_dataset

# Browse available OWAMcap datasets
datasets = load_dataset.list_available(format="OWA")

# Load a specific dataset
data = load_dataset("open-world-agents/example_dataset")
```

**Available Datasets**: [🤗 datasets?other=OWA](https://huggingface.co/datasets?other=OWA)

### Data Format Preview

**OWAMcap** combines the robustness of [MCAP](https://mcap.dev/) with specialized desktop interaction schemas. Perfect synchronization of screen captures, input events, and window context with nanosecond precision.

**Key Features**:
- 🔄 **Universal Standard**: Unlike fragmented formats, enables seamless dataset combination for large-scale foundation models *(OWAMcap)*
- 🎯 **High-Performance Multimodal Storage**: Lightweight [MCAP](https://mcap.dev/) container with nanosecond precision for synchronized data streams *(MCAP)*
- 🔗 **Flexible MediaRef**: Smart references to both external and embedded media (file paths, URLs, data URIs, video frames) with lazy loading - keeps metadata files small while supporting rich media *(OWAMcap)* → [Learn more](https://open-world-agents.github.io/open-world-agents/data/technical-reference/format-guide/#media-handling)
- 🤗 **Training Pipeline Ready**: Native HuggingFace integration, seamless dataset loading, and direct compatibility with ML frameworks *(Ecosystem)* → [Browse datasets](https://huggingface.co/datasets?other=OWA) | [Data pipeline](https://open-world-agents.github.io/open-world-agents/data/technical-reference/data-pipeline/)

> 📖 **Learn More**: [Why OWAMcap?](https://open-world-agents.github.io/open-world-agents/data/getting-started/why-owamcap/) | [Complete Format Guide](https://open-world-agents.github.io/open-world-agents/data/technical-reference/format-guide/) | [vs Other Formats](https://open-world-agents.github.io/open-world-agents/data/tools/comparison-with-lerobot/)

```bash
$ owl mcap info example.mcap
library:   mcap-owa-support 0.5.1; mcap 1.3.0
profile:   owa
messages:  864
duration:  10.3574349s
start:     2025-06-27T18:49:52.129876+09:00 (1751017792.129876000)
end:       2025-06-27T18:50:02.4873109+09:00 (1751017802.487310900)
compression:
        zstd: [1/1 chunks] [116.46 KiB/16.61 KiB (85.74%)] [1.60 KiB/sec]
channels:
        (1) window           11 msgs (1.06 Hz)    : desktop/WindowInfo [jsonschema]
        (2) keyboard/state   11 msgs (1.06 Hz)    : desktop/KeyboardState [jsonschema]
        (3) mouse/state      11 msgs (1.06 Hz)    : desktop/MouseState [jsonschema]
        (4) screen          590 msgs (56.96 Hz)   : desktop/ScreenCaptured [jsonschema]
        (5) mouse           209 msgs (20.18 Hz)   : desktop/MouseEvent [jsonschema]
        (6) keyboard         32 msgs (3.09 Hz)    : desktop/KeyboardEvent [jsonschema]
channels: 6
attachments: 0
metadata: 0
```

## 🛠️ CLI Tools (`owl`)

```bash
# Data analysis
owl mcap info session.mcap              # File overview & statistics
owl mcap cat session.mcap --n 10        # View messages
owl video probe session.mkv             # Video analysis

# Environment management
owl env list                            # List plugins
owl env search "mouse.*click"           # Search components
owl messages show desktop/KeyboardEvent # View schemas
```

> **💡 Complete CLI Reference**: For detailed information about all CLI commands and options, see the [CLI Tools documentation](https://open-world-agents.github.io/open-world-agents/cli).

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

For development or contributing to the project, you can install packages in editable mode. For detailed development setup instructions, see the [Installation Guide](https://open-world-agents.github.io/open-world-agents/install/).


## Features

- **🔄 Asynchronous Framework**: Real-time event handling with Callables, Listeners, and Runnables
- **🧩 Zero-Configuration Plugin System**: Automatic plugin discovery via Entry Points
- **🎥 High-Performance Environment Plugins**: 6x faster screen capture via `owa.env.gst` GStreamer integration
- **🗂️ Universal Data Format**: [OWAMcap standard](https://open-world-agents.github.io/open-world-agents/data/getting-started/why-owamcap/) with flexible MediaRef system for efficient external media references
- **🤗 HuggingFace Ecosystem**: Growing collection of community datasets, plugins, and tools
- **🛠️ Extensible Architecture**: Modular design for custom environments and message types

## Documentation

- **Full Documentation**: https://open-world-agents.github.io/open-world-agents/
- **Environment Framework**: [Environment Guide](https://open-world-agents.github.io/open-world-agents/env/) - Core concepts, usage guide, and plugin development
- **Data Infrastructure**: [Data Guide](https://open-world-agents.github.io/open-world-agents/data/) - Recording, storage, and analysis with [OWAMcap format](https://open-world-agents.github.io/open-world-agents/data/getting-started/why-owamcap/)
- **CLI Tools**: [CLI Reference](https://open-world-agents.github.io/open-world-agents/cli/) - Command-line utilities and reference

## Contributing

We welcome contributions! Whether you're:
- Building new environment plugins
- Improving performance
- Adding documentation
- Reporting bugs

Please see our [Contributing Guide](https://open-world-agents.github.io/open-world-agents/contributing/) for details.

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.
