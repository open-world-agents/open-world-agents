<div align="center">
  <img src="images/owa-logo.jpg" alt="Open World Agents" width="300"/>
</div>

# Open World Agents Documentation

Open World Agents (OWA) is a monorepo for building AI agents that interact with desktop applications. It provides data capture, environment control, and training utilities.

## Quick Start

<!-- SYNC-ID: quick-start-3-steps -->
```bash
# 1. Record desktop interaction
$ ocap my-session.mcap

# 2. Process to training format
$ python scripts/01_raw_to_event.py --train-dir ./

# 3. Train your model
$ python train.py --dataset ./event-dataset
```
<!-- END-SYNC: quick-start-3-steps -->

> 📖 **Detailed Guide**: [Complete Quick Start Tutorial](quick-start.md)

## Architecture Overview

OWA consists of the following core components:

<!-- SYNC-ID: core-components-list -->
- 🌍 **[Environment Framework](env/index.md)**: "USB-C of desktop agents" - universal interface for native desktop automation with pre-built plugins for desktop control, high-performance screen capture, and zero-configuration plugin system
- 📊 **[Data Infrastructure](data/index.md)**: Complete desktop agent data pipeline from recording to training with `OWAMcap` format - a [universal standard](data/getting-started/why-owamcap.md) powered by [MCAP](https://mcap.dev/)
- 🛠️ **[CLI Tools](cli/index.md)**: Command-line utilities (`owl`) for recording, analyzing, and managing agent data
- 🤖 **[Examples](examples/index.md)**: Complete implementations and training pipelines for multimodal agents
<!-- END-SYNC: core-components-list -->

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

The easiest way to get started is to install the [**owa**](https://github.com/open-world-agents/open-world-agents/blob/main/pyproject.toml) meta-package, which includes all core components and environment plugins:

```bash
$ pip install owa
```

All OWA packages use namespace packaging and are installed in the `owa` namespace (e.g., `owa.core`, `owa.cli`, `owa.env.desktop`). For more detail, see [Packaging namespace packages](https://packaging.python.org/en/latest/guides/packaging-namespace-packages/). We recommend using [`uv`](https://docs.astral.sh/uv/) as the package manager.

| Name | PyPI | Conda | Description |
|------|------|-------|-------------|
| [`owa.core`](https://github.com/open-world-agents/open-world-agents/tree/main/projects/owa-core) | [![owa-core](https://img.shields.io/pypi/v/owa-core?label=owa-core)](https://pypi.org/project/owa-core/) | [![owa-core](https://img.shields.io/conda/vn/conda-forge/owa-core?label=conda)](https://anaconda.org/conda-forge/owa-core) | Framework foundation with registry system |
| [`owa.msgs`](https://github.com/open-world-agents/open-world-agents/tree/main/projects/owa-msgs) | [![owa-msgs](https://img.shields.io/pypi/v/owa-msgs?label=owa-msgs)](https://pypi.org/project/owa-msgs/) | [![owa-msgs](https://img.shields.io/conda/vn/conda-forge/owa-msgs?label=conda)](https://anaconda.org/conda-forge/owa-msgs) | Core message definitions with automatic discovery |
| [`owa.cli`](https://github.com/open-world-agents/open-world-agents/tree/main/projects/owa-cli) | [![owa-cli](https://img.shields.io/pypi/v/owa-cli?label=owa-cli)](https://pypi.org/project/owa-cli/) | [![owa-cli](https://img.shields.io/conda/vn/conda-forge/owa-cli?label=conda)](https://anaconda.org/conda-forge/owa-cli) | Command-line tools (`owl`) for data analysis |
| [`mcap-owa-support`](https://github.com/open-world-agents/open-world-agents/tree/main/projects/mcap-owa-support) | [![mcap-owa-support](https://img.shields.io/pypi/v/mcap-owa-support?label=mcap-owa-support)](https://pypi.org/project/mcap-owa-support/) | [![mcap-owa-support](https://img.shields.io/conda/vn/conda-forge/mcap-owa-support?label=conda)](https://anaconda.org/conda-forge/mcap-owa-support) | OWAMcap format support and utilities |
| [`ocap`](https://github.com/open-world-agents/ocap) 🎥 | [![ocap](https://img.shields.io/pypi/v/ocap?label=ocap)](https://pypi.org/project/ocap/) | [![ocap](https://img.shields.io/conda/vn/conda-forge/ocap?label=conda)](https://anaconda.org/conda-forge/ocap) | Desktop recorder for multimodal data capture |
| [`owa.env.desktop`](https://github.com/open-world-agents/open-world-agents/tree/main/projects/owa-env-desktop) | [![owa-env-desktop](https://img.shields.io/pypi/v/owa-env-desktop?label=owa-env-desktop)](https://pypi.org/project/owa-env-desktop/) | [![owa-env-desktop](https://img.shields.io/conda/vn/conda-forge/owa-env-desktop?label=conda)](https://anaconda.org/conda-forge/owa-env-desktop) | Mouse, keyboard, window event handling |
| [`owa.env.gst`](https://github.com/open-world-agents/open-world-agents/tree/main/projects/owa-env-gst) 🎥 | [![owa-env-gst](https://img.shields.io/pypi/v/owa-env-gst?label=owa-env-gst)](https://pypi.org/project/owa-env-gst/) | [![owa-env-gst](https://img.shields.io/conda/vn/conda-forge/owa-env-gst?label=conda)](https://anaconda.org/conda-forge/owa-env-gst) | High-performance, hardware-accelerated screen capture |
| [`owa.env.example`](https://github.com/open-world-agents/open-world-agents/tree/main/projects/owa-env-example) | - | - | Reference implementations for learning |

> 🎥 **Video Processing Packages**: Packages marked with 🎥 require GStreamer dependencies. Install `$ conda install open-world-agents::gstreamer-bundle` first for full functionality.

> 📦 **Lockstep Versioning**: All first-party OWA packages follow lockstep versioning, meaning they share the same version number to ensure compatibility and simplify dependency management.

---

## 🌍 Environment Framework

Universal interface for native desktop automation with real-time event handling and zero-configuration plugin discovery.

### Environment Navigation

| Section | Description |
|---------|-------------|
| **[Environment Overview](env/index.md)** | Core concepts and quick start guide |
| **[Environment Guide](env/guide.md)** | Complete system overview and usage examples |
| **[Custom Plugins](env/custom_plugins.md)** | Create your own environment extensions |
| **[CLI Tools](cli/env.md)** | Plugin management and exploration commands |

**Built-in Plugins:**

| Plugin | Description | Key Features |
|--------|-------------|--------------|
| **[Standard](env/plugins/std.md)** | Core utilities | Time functions, periodic tasks |
| **[Desktop](env/plugins/desktop.md)** | Desktop automation | Mouse/keyboard control, window management |
| **[GStreamer](env/plugins/gst.md)** | Hardware-accelerated capture | Fast screen recording |

---

## 📊 Data Infrastructure

Desktop AI needs high-quality, synchronized multimodal data: screen captures, mouse/keyboard events, and window context. OWA provides the **complete pipeline** from recording to training.

### 🚀 Getting Started
New to OWA data? Start here:

- **[Why OWAMcap?](data/getting-started/why-owamcap.md)** - Understand the problem and solution
- **[Recording Data](data/getting-started/recording-data.md)** - Capture desktop interactions with `ocap`
- **[Exploring Data](data/getting-started/exploring-data.md)** - View and analyze your recordings

### 📚 Technical Reference

- **[OWAMcap Format Guide](data/technical-reference/format-guide.md)** - Complete technical specification
- **[Data Pipeline](data/technical-reference/data-pipeline.md)** - Transform recordings to training-ready datasets

### 🛠️ Tools & Ecosystem

- **[Data Viewer](data/viewer.md)** - Web-based visualization tool
- **[Data Conversions](data/conversions.md)** - Convert existing datasets (VPT, CS:GO) to OWAMcap
- **[CLI Tools (owl)](cli/index.md)** - Command-line interface for data analysis and management

### 🤗 Community Datasets

<!-- SYNC-ID: community-datasets -->
**Browse Datasets**: [🤗 HuggingFace](https://huggingface.co/datasets?other=OWA)

- **Standardized Format**: All datasets use OWAMcap for seamless integration
- **Interactive Preview**: [Hugging Face Spaces Visualizer](https://huggingface.co/spaces/open-world-agents/visualize_dataset)
<!-- END-SYNC: community-datasets -->

---

## 🤖 Examples

| Example | Description | Status |
|---------|-------------|---------|
| **[Multimodal Game Agent](examples/multimodal_game_agent.md)** | Vision-based game playing agent | 🚧 In Progress |
| **[GUI Agent](examples/gui_agent.md)** | General desktop application automation | 🚧 In Progress |
| **[Interactive World Model](examples/interactive_world_model.md)** | Predictive modeling of desktop environments | 🚧 In Progress |
| **[Usage with LLMs](examples/usage_with_llm.md)** | Integration with large language models | 🚧 In Progress |
| **[Usage with Transformers](examples/usage_with_transformers.md)** | Vision transformer implementations | 🚧 In Progress |

## Development Resources
Learn how to contribute, report issues, and get help.

| Resource | Description |
|----------|-------------|
| **[Help with OWA](help_with_owa.md)** | Community support resources |
| **[Installation Guide](install.md)** | Detailed installation instructions |
| **[Contributing Guide](contributing.md)** | Development setup, bug reports, feature proposals |
| **[FAQ for Developers](faq_dev.md)** | Common questions and troubleshooting |

---

## Features

### 🌍 Environment Framework: "USB-C of Desktop Agents"
<!-- SYNC-ID: env-framework-features -->
- **⚡ Real-time Performance**: Optimized for responsive agent interactions (GStreamer components achieve <30ms latency)
- **🔌 Zero-Configuration**: Automatic plugin discovery via Python Entry Points
- **🌐 Event-Driven**: Asynchronous processing that mirrors real-world dynamics
- **🧩 Extensible**: Community-driven plugin ecosystem
<!-- END-SYNC: env-framework-features -->

[**→ View Environment Framework Guide**](env/index.md)

### 📊 Data Infrastructure: Complete Pipeline

<!-- SYNC-ID: owamcap-key-features -->
- 🌐 **Universal Standard**: Unlike fragmented formats, enables seamless dataset combination for large-scale foundation models *(OWAMcap)*
- ⚡ **High-Performance Multimodal Storage**: Lightweight [MCAP](https://mcap.dev/) container with nanosecond precision for synchronized data streams *(MCAP)*
- 🔗 **Flexible MediaRef**: Smart references to both external and embedded media (file paths, URLs, data URIs, video frames) with lazy loading - keeps metadata files small while supporting rich media *(OWAMcap)* → [Learn more](data/technical-reference/format-guide.md#media-handling)
- 🤗 **Training Pipeline Ready**: Native HuggingFace integration, seamless dataset loading, and direct compatibility with ML frameworks *(Ecosystem)* → [Browse datasets](https://huggingface.co/datasets?other=OWA) | [Data pipeline](data/technical-reference/data-pipeline.md)
<!-- END-SYNC: owamcap-key-features -->

[**→ View Data Infrastructure Guide**](data/index.md)

### 🤗 Community & Ecosystem

- **🌱 Growing Ecosystem**: Hundreds of community datasets in unified OWAMcap format
- **🤗 HuggingFace Integration**: Native dataset loading, sharing, and interactive preview tools
- **🧩 Extensible Architecture**: Modular design for custom environments, plugins, and message types
- **💡 Community-Driven**: Plugin ecosystem spanning gaming, web automation, mobile control, and specialized domains

[**→ View Community Datasets**](https://huggingface.co/datasets?other=OWA)

---

## License

This project is released under the MIT License. See the [LICENSE](https://github.com/open-world-agents/open-world-agents/blob/main/LICENSE) file for details.
