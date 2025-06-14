<div align="center">
  <img src="images/owa-logo.jpg" alt="Open World Agents" width="300"/>
</div>

# Open World Agents Documentation

**A comprehensive framework for building AI agents that interact with desktop applications through vision, keyboard, and mouse control.**

Open World Agents (OWA) is a monorepo containing the complete toolkit for multimodal desktop agent development. From high-performance data capture to model training and real-time evaluation, everything is designed for flexibility and performance.

## Architecture Overview

OWA consists of four core components:

🌍 **[Environment (Env)](env/index.md)** - Asynchronous, event-driven interface for real-time agent interactions
📊 **[Data](data/index.md)** - High-performance recording, storage, and analysis of multimodal desktop data
📝 **[Messages](env/guide.md#message-registry)** - Centralized message definitions with automatic discovery and registry system
🤖 **[Examples](examples/)** - Complete implementations and training pipelines for multimodal agents

## Quick Navigation

### 🌍 Environment Framework
Build reactive desktop agents with our asynchronous environment system.

| Component | Description |
|-----------|-------------|
| **[Core Concepts](env/index.md)** | `Callables`, `Listeners`, and `Runnables` for real-time processing |
| **[Environment Guide](env/guide.md)** | Zero-configuration plugin system and CLI tools |
| **[Custom Plugins](env/custom_plugins.md)** | Create your own environment extensions |

**Built-in Plugins:**  

- **[Desktop Environment](env/plugins/desktop.md)** - Mouse, keyboard, and window event handling
- **[GStreamer Environment](env/plugins/gst.md)** - High-performance screen capture (**6x faster** than alternatives)
- **[Standard Environment](env/plugins/std.md)** - Basic utilities and timing functions

### 📊 Data Infrastructure
Capture, store, and analyze multimodal desktop interaction data.

| Component | Description |
|-----------|-------------|
| **[Data Overview](data/index.md)** | Complete data pipeline for desktop agents |
| **[OWAMcap Format](data/data_format.md)** | Specialized format capturing complete desktop interactions (screen + events) with nanosecond precision |
| **[Desktop Recorder (ocap)](data/ocap.md)** | High-performance desktop recording tool |
| **[Data Viewer](data/viewer.md)** | Visualize and analyze recorded sessions |
| **[Data Explorer](data/how_to_explorer_and_edit.md)** | Tools for data exploration and editing |

### 🤖 Agent Examples
Learn from complete implementations and training pipelines.

| Example | Description | Status |
|---------|-------------|---------|
| **[Multimodal Game Agent](examples/multimodal_game_agent.md)** | Vision-based game playing agent | 🚧 In Progress |
| **[GUI Agent](examples/gui_agent.md)** | General desktop application automation | 🚧 In Progress |
| **[Interactive World Model](examples/interactive_world_model.md)** | Predictive modeling of desktop environments | 🚧 In Progress |
| **[Usage with LLMs](examples/usage_with_llm.md)** | Integration with large language models | 🚧 In Progress |
| **[Usage with Transformers](examples/usage_with_transformers.md)** | Vision transformer implementations | 🚧 In Progress |

## Community & Ecosystem

**🌱 Growing Ecosystem**: OWA is designed for extensibility. Community contributions include:  

- Custom environment plugins (`owa.env.minecraft`, `owa.env.web`, etc.)  
- Specialized data processors and analyzers  
- Novel agent architectures and training methods  

**🤗 HuggingFace Integration**: Upload and share datasets created with `ocap`. Preview datasets at [HuggingFace Spaces](https://huggingface.co/spaces/open-world-agents/visualize_dataset).

## Development Resources

| Resource | Description |
|----------|-------------|
| **[Installation Guide](install.md)** | Detailed installation instructions |
| **[Contributing Guide](contributing.md)** | Development setup, bug reports, feature proposals |
| **[FAQ](faq_dev.md)** | Common questions and troubleshooting |

---

## What Can You Build?

**Anything that runs on desktop.** If a human can do it on a computer, you can build an AI agent to automate it:

🤖 **Desktop Automation** - Navigate applications, automate workflows, interact with any software  
🎮 **Game AI** - Master complex games through visual understanding and real-time decision making  
📊 **Training Datasets** - Capture high-quality human-computer interaction data for foundation models  
📈 **Benchmarks** - Create and evaluate desktop agent performance across diverse tasks

## License

This project is released under the MIT License. See the [LICENSE](https://github.com/open-world-agents/open-world-agents/blob/main/LICENSE) file for details.
