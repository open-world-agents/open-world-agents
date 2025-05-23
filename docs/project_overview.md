<div align="center">
  <img src="/images/owa-logo.jpg" alt="OWA Project Overview" width="120"/>
</div>

# 🏗️ Project Overview

**Open World Agents (OWA)** is a comprehensive mono-repository architecture designed for building state-of-the-art multimodal desktop agents.

## 📁 Repository Structure

`open-world-agents` is a mono-repo which is composed with multiple sub-repository. e.g. `projects/mcap-owa-support, projects/owa-cli, projects/owa-core, projects/owa-env-desktop`.

Each sub-repository is a self-contained repository which may have other sub-repository as dependencies.

Most of subprojects inside `projects` are **python package in itself**; In other words, they are installable by `pip` or [`uv`](https://docs.astral.sh/uv/). Since we're utilizing `uv`, we recommend you to use `uv` as package manager.

## 🎯 Namespace Package Architecture

We're adopting namespace packages for clean, modular organization. Most `owa`-related packages, including EnvPlugins, are installed in `owa` namespace, e.g. `owa.core`, `owa.cli`, `owa.env.desktop`. For more detail, see [Packaging namespace packages](https://packaging.python.org/en/latest/guides/packaging-namespace-packages/)

```
open-world-agents/
├── projects/
│   ├── mcap-owa-support    # 📦 MCAP format support library
│   ├── owa-core/           # 🏗️ Core framework & registry system
│   ├── owa-cli/            # 🖥️ Command-line interface tools
│   ├── owa-env-desktop/    # 🖱️ Desktop interaction plugins
│   ├── owa-env-example/    # 📚 Example environment plugins
│   ├── owa-env-gst/        # 🎬 GStreamer-based capture engine
│   └── and also more! e.g. you may contribute owa-env-minecraft!
├── docs/                   # 📖 Documentation (this site!)
└── README.md              # 🚀 Project overview
```

## 🔗 Integration & Extensibility

The modular design enables:

- **Mix & Match**: Use only the components you need
- **Easy Extension**: Add custom plugins following our patterns
- **Community Driven**: Contribute new environment plugins like `owa-env-minecraft`!
- **Production Ready**: Each component is independently testable and deployable
