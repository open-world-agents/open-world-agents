<div align="center">
  <img src="images/owa-logo.jpg" alt="Open World Agents" width="300"/>
</div>

# Open World Agents Documentation

**A comprehensive framework for building AI agents that interact with desktop applications through vision, keyboard, and mouse control.**

Open World Agents (OWA) is a monorepo containing the complete toolkit for multimodal desktop agent development. From high-performance data capture to model training and real-time evaluation, everything is designed for flexibility and performance.

## 🚀 Quick Start: Record → Train in 3 Steps

<!-- SYNC-ID: quick-start-3-steps -->
```bash
# 1. Record desktop interaction
$ ocap my-session.mcap

# 2. Process to training format
$ python scripts/01_raw_events_to_event_dataset.py --train-dir ./

# 3. Train your model
$ python train.py --dataset ./event-dataset
```

> 📖 **Detailed Guide**: [Complete Quick Start Tutorial](quick-start.md) - Step-by-step walkthrough with examples and troubleshooting
<!-- END-SYNC: quick-start-3-steps -->

## Architecture Overview

OWA consists of the following core components:

<!-- SYNC-ID: core-components-list -->
- 🌍 **[Environment Framework](env/index.md)** - Universal interface for native desktop automation ("USB-C of desktop agents") with pre-built plugins for desktop control, high-performance screen capture (6x faster), and zero-configuration plugin system
- 📊 **[Data Infrastructure](data/index.md)** - Complete desktop agent data pipeline from recording to training with `OWAMcap` format - a [universal standard](data/getting-started/why-owamcap.md) powered by [mcap](https://mcap.dev/)
- 🛠️ **[CLI Tools](cli/index.md)** - Command-line utilities (`owl`) for recording, analyzing, and managing agent data
- 🤖 **[Examples](examples/)** - Complete implementations and training pipelines for multimodal agents
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
| **[GStreamer](env/plugins/gst.md)** | High-performance capture | 6x faster screen recording |

---

## 📊 Data Infrastructure: Complete Desktop Agent Data Pipeline

Desktop AI needs high-quality, synchronized multimodal data: screen captures, mouse/keyboard events, and window context. OWA provides the **complete pipeline** from recording to training.

### The OWA Data Ecosystem

**🎯 Getting Started**
New to OWA data? Start here:

- **[Why OWAMcap?](data/getting-started/why-owamcap.md)** - Understand the problem and solution
- **[Recording Data](data/getting-started/recording-data.md)** - Capture desktop interactions with `ocap`
- **[Exploring Data](data/getting-started/exploring-data.md)** - View and analyze your recordings

**📚 Technical Reference**
Deep dive into the format and pipeline:

- **[OWAMcap Format Guide](data/technical-reference/format-guide.md)** - Complete technical specification
- **[Data Pipeline](data/technical-reference/data-pipeline.md)** - Transform recordings to training-ready datasets

**🛠️ Tools & Ecosystem**

- **[Data Viewer](data/tools/viewer.md)** - Web-based visualization tool
- **[Comparison with LeRobot](data/tools/comparison-with-lerobot.md)** - Technical comparison with alternatives
- **[CLI Tools (owl)](cli/index.md)** - Command-line interface for data analysis and management

### 🤗 Community Datasets

<!-- SYNC-ID: community-datasets -->
**Browse Available Datasets**: [🤗 datasets?other=OWA](https://huggingface.co/datasets?other=OWA)

- **Growing Collection**: Hundreds of community-contributed datasets
- **Standardized Format**: All use OWAMcap for seamless integration
- **Interactive Preview**: [Hugging Face Spaces Visualizer](https://huggingface.co/spaces/open-world-agents/visualize_dataset)
- **Easy Sharing**: Upload recordings directly with one command

> 🚀 **Impact**: OWA has democratized desktop agent data, growing from zero to hundreds of public datasets in the unified OWAMcap format.
<!-- END-SYNC: community-datasets -->

---

## 🤖 Awesome Examples
Learn from complete implementations and training pipelines.

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
