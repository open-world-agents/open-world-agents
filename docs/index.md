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
$ python scripts/01_raw_events_to_event_dataset.py --train-dir ./

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

## License

This project is released under the MIT License. See the [LICENSE](https://github.com/open-world-agents/open-world-agents/blob/main/LICENSE) file for details.
