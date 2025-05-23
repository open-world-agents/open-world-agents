<div align="center">
  <img src="docs/images/owa-logo.jpg" alt="Open World Agents" width="300"/>
  
  # 🚀 Open World Agents
  
  **Everything you need to build state-of-the-art foundation multimodal desktop agent, end-to-end.**
  
  [![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://open-world-agents.github.io/open-world-agents/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
  [![GitHub stars](https://img.shields.io/github/stars/open-world-agents/open-world-agents?style=social)](https://github.com/open-world-agents/open-world-agents/stargazers)
  
</div>

---

## 🌟 What is Open World Agents?

Open World Agents (OWA) is a **comprehensive framework** for building, training, and deploying multimodal desktop agents that can interact with real-world applications in real-time. Unlike traditional RL environments that operate in simulated worlds, OWA enables agents to learn and operate directly on actual desktop environments with **sub-30ms latency**.

### 🎯 Key Capabilities

- **🎬 Real-Time Screen Capture**: Hardware-accelerated H.265 encoding with DirectX 11 + GStreamer
- **🖱️ Precise Input Simulation**: Sub-millisecond mouse, keyboard, and window interactions
- **📊 Multimodal Data Recording**: Synchronized screen, audio, input events in efficient MCAP format
- **🤖 Agent Training Pipeline**: End-to-end data collection → model training → deployment workflow
- **⚡ Production Performance**: 6x faster than alternatives with <30ms end-to-end latency

### 🏗️ Why OWA?

Traditional agent frameworks like OpenAI Gym assume infinite processing time with `env.step()`. **Real-world agents need to react instantly**—just like humans interacting with computers. OWA's asynchronous, event-driven architecture enables true real-time agent interactions.

## 📦 Repository Structure

OWA is organized as a **monorepo** with multiple independent Python packages, following **namespace package** conventions for clean modularity:

```
open-world-agents/
├── projects/
│   ├── owa-core/           # 🏗️  Core framework & registry system
│   ├── owa-env-desktop/    # 🖱️  Desktop interaction plugins
│   ├── owa-env-gst/        # 🎬  GStreamer-based capture engine
│   ├── owa-cli/            # 🖥️  Command-line interface tools
│   ├── ocap/               # 📹  Desktop recorder application
│   ├── mcap-owa-support/   # 📦  MCAP format support library
│   └── owa-mcap-viewer/    # 👁️  Data visualization tools
├── docs/                   # 📖  Documentation website
└── examples/               # 🧪  Example agents & tutorials
```

## 🚀 Quick Start

### 📥 Installation

**Option 1: Full Installation (Recommended)**

```bash
# Clone the repository
git clone https://github.com/open-world-agents/open-world-agents
cd open-world-agents

# Install with uv (recommended)
uv sync --inexact

# Or with pip
pip install -e projects/owa-core -e projects/owa-cli -e projects/ocap
```

**Option 2: PyPI Installation (Experimental)**

```bash
pip install owa
```

> **⚠️ Platform Requirements**: Currently supports **Windows 10/11** with **NVIDIA GPU** for optimal performance. macOS/Linux support planned.

### 🎬 Record Your First Dataset

```bash
# Start recording desktop interactions
ocap my-first-recording.mcap

# Interact with your desktop...
# Press Ctrl+C to stop

# Explore the recorded data
ocap info my-first-recording.mcap
ocap cat my-first-recording.mcap --n 10
```

### 🤖 Build Your First Agent

```python
import time
from owa.core.registry import CALLABLES, LISTENERS, activate_module

# Activate desktop environment
activate_module("owa.env.desktop")
activate_module("owa.env.gst")

def on_screen_update(frame, metrics):
    print(f"📸 New frame: {frame.frame_arr.shape}")
    print(f"⚡ Latency: {metrics.latency*1000:.1f}ms")

# Start real-time screen capture
screen = LISTENERS["screen"]().configure(
    callback=on_screen_update,
    fps=60,
    show_cursor=True
)

with screen.session:
    print("🎯 Agent is watching your screen...")
    time.sleep(5)
```

## 📚 Package Ecosystem

### Core Framework

| Package                           | PyPI       | Description                                           | Installation           |
| --------------------------------- | ---------- | ----------------------------------------------------- | ---------------------- |
| **[owa-core](projects/owa-core)** | `owa-core` | 🏗️ Core registry system, base classes, and utilities  | `pip install owa-core` |
| **[owa-cli](projects/owa-cli)**   | `owa-cli`  | 🖥️ Command-line tools for data analysis and debugging | `pip install owa-cli`  |

### Environment Plugins

| Package                                         | PyPI              | Description                                       | Installation                  |
| ----------------------------------------------- | ----------------- | ------------------------------------------------- | ----------------------------- |
| **[owa-env-desktop](projects/owa-env-desktop)** | `owa-env-desktop` | 🖱️ Mouse, keyboard, window interaction handlers   | `pip install owa-env-desktop` |
| **[owa-env-gst](projects/owa-env-gst)**         | `owa-env-gst`     | 🎬 Hardware-accelerated screen capture with H.265 | Requires GStreamer + conda    |

### Data & Recording

| Package                                           | PyPI               | Description                          | Installation                   |
| ------------------------------------------------- | ------------------ | ------------------------------------ | ------------------------------ |
| **[ocap](projects/ocap)**                         | `ocap`             | 📹 High-performance desktop recorder | `pip install ocap`             |
| **[mcap-owa-support](projects/mcap-owa-support)** | `mcap-owa-support` | 📦 MCAP format encoders/decoders     | `pip install mcap-owa-support` |
| **[owa-mcap-viewer](projects/owa-mcap-viewer)**   | `owa-mcap-viewer`  | 👁️ Interactive data visualization    | `pip install owa-mcap-viewer`  |

### Experimental

| Package                                         | Description                            | Status         |
| ----------------------------------------------- | -------------------------------------- | -------------- |
| **[owa-env-example](projects/owa-env-example)** | 📚 Example environment implementations | 🧪 Development |

> **💡 Pro Tip**: Install the `owa` meta-package to get core components automatically. For specialized needs, install individual packages.

## 🏗️ Architecture Overview

OWA follows an **event-driven, plugin-based architecture** that separates concerns and enables modular development:

### 🔌 Plugin System

- **Callables**: Synchronous function calls (`mouse.click()`, `window.focus()`)
- **Listeners**: Asynchronous event handlers (screen capture, keyboard monitoring)
- **Runnables**: Long-running background processes (continuous recording)

**📖 Learn More**: [Environment System Guide](https://open-world-agents.github.io/open-world-agents/env/) | [Plugin Development](https://open-world-agents.github.io/open-world-agents/env/custom_plugins/) | [Core Implementation](projects/owa-core/)

### 📊 Data Pipeline

1. **Capture**: Real-time multimodal data collection
2. **Storage**: Efficient MCAP format with ZSTD compression
3. **Processing**: Event synchronization and analysis
4. **Training**: Integration with ML frameworks
5. **Deployment**: Production agent execution

**📖 Learn More**: [Data Formats](https://open-world-agents.github.io/open-world-agents/data/data_format/) | [MCAP Specification](https://open-world-agents.github.io/open-world-agents/data/ocap/) | [Recording Tools](projects/ocap/)

### ⚡ Performance Optimizations

- **Hardware Acceleration**: DirectX 11 + NVIDIA GPU encoding
- **Zero-Copy Operations**: Minimize memory allocations
- **Asynchronous Processing**: Non-blocking event handling
- **Efficient Compression**: 80%+ size reduction with ZSTD

**📖 Learn More**: [GStreamer Engine](https://open-world-agents.github.io/open-world-agents/env/plugins/gstreamer_env/) | [Performance Benchmarks](https://open-world-agents.github.io/open-world-agents/env/plugins/gstreamer_env/#performance-advantages) | [Technical Deep Dive](projects/owa-env-gst/)

### 🛠️ Implementation Details

- **Registry Pattern**: [Core Registry System](projects/owa-core/owa/core/registry.py) - Dynamic plugin activation and management
- **Message System**: [Base Message Classes](projects/owa-core/owa/core/message.py) - Standardized event communication
- **Screen Capture**: [Hardware-Accelerated Pipeline](projects/owa-env-gst/owa/env/gst/pipeline_builder/) - DirectX 11 + H.265 encoding
- **Desktop Integration**: [Windows API Wrappers](projects/owa-env-desktop/) - Mouse, keyboard, window control

## 🎓 Learn More

### 📖 Documentation

- **[Getting Started Guide](https://open-world-agents.github.io/open-world-agents/)**
- **[Environment System](https://open-world-agents.github.io/open-world-agents/env/)**
- **[Data Formats](https://open-world-agents.github.io/open-world-agents/data/)**
- **[API Reference](https://open-world-agents.github.io/open-world-agents/api/)**

### 🧪 Examples & Tutorials

- **Desktop Game Agent**: Train an agent to play browser games
- **GUI Automation**: Automate complex desktop workflows
- **Data Collection**: Build custom recording pipelines
- **Real-time Inference**: Deploy agents in production

### 🤝 Community

- **[GitHub Issues](https://github.com/open-world-agents/open-world-agents/issues)**: Bug reports and feature requests
- **[Discussions](https://github.com/open-world-agents/open-world-agents/discussions)**: Community Q&A
- **[Contributing Guide](https://open-world-agents.github.io/open-world-agents/contributing/)**: How to contribute

## 🔄 Versioning

We follow [Semantic Versioning](https://semver.org/) (SemVer) for all packages:

- **0.x.x**: Development phase - breaking changes may occur
- **1.x.x**: Stable API - breaking changes only on major versions
- **Lockstep Versioning**: All first-party packages share the same version number

> **⚠️ API Stability**: Modules starting with underscore `_` are internal APIs and should not be imported directly. Only use public APIs from `owa.*` namespaces.

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### 🐛 Report Issues

Found a bug? Have a feature request? [Open an issue](https://github.com/open-world-agents/open-world-agents/issues/new)!

### 🔧 Development Setup

```bash
# Fork the repository on GitHub
git clone https://github.com/YOUR-USERNAME/open-world-agents
cd open-world-agents

# Set up development environment
uv sync --all-extras --dev
pre-commit install

# Run tests
pytest
```

### 📝 Areas for Contribution

- **🌍 Platform Support**: macOS/Linux compatibility
- **🎮 Environment Plugins**: New application integrations
- **📊 ML Integrations**: Training pipeline improvements
- **📖 Documentation**: Tutorials and examples
- **🧪 Testing**: Test coverage and CI/CD

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🏆 Acknowledgments

OWA builds upon excellent open-source technologies:

- **[GStreamer](https://gstreamer.freedesktop.org/)**: Multimedia processing framework
- **[MCAP](https://mcap.dev/)**: Modular container format for multimodal data
- **[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)**: Documentation framework

---

<div align="center">
  
  **Ready to build the future of desktop agents?**
  
  🚀 **[Get Started Now](https://open-world-agents.github.io/open-world-agents/)** | 📖 **[Read the Docs](https://open-world-agents.github.io/open-world-agents/)** | 💬 **[Join Discussions](https://github.com/open-world-agents/open-world-agents/discussions)**
  
  *Star ⭐ this repository if you find it useful!*
  
</div>
