# Project Overview

Open World Agents is a comprehensive framework for building AI agents that interact with desktop applications through vision, keyboard, and mouse control—a complete toolkit from data capture to model training and evaluation.

> **📄 Research Paper**: This project was developed for the [D2E project](https://worv-ai.github.io/d2e/). If you find this work useful, please cite our paper.

## Why OWA?

**Fragmented tools make desktop AI development painful.** Most solutions force you to:

- Stitch together incompatible recording tools
- Build custom data pipelines from scratch
- Handle real-time performance issues yourself
- Start agent development with no examples

**OWA solves this** with a unified framework: record with [`ocap`](https://github.com/open-world-agents/ocap), train with standardized datasets, deploy with real-time environment components, and learn from community examples.

## What Can You Build?

- 🤖 **Desktop Automation**: Navigate applications, automate workflows, interact with any software
- 🎮 **Game AI**: Master complex games through visual understanding and real-time decision making
- 📊 **Training Datasets**: Capture high-quality human-computer interaction data for foundation models
- 🤗 **Community Datasets**: Access and contribute to growing [OWAMcap datasets](https://huggingface.co/datasets?other=OWA) on HuggingFace
- 📈 **Benchmarks**: Create and evaluate desktop agent performance across diverse tasks

## Core Components

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
| [`owa.env.gst`](https://github.com/open-world-agents/open-world-agents/tree/main/projects/owa-env-gst) 🎥 | [![owa-env-gst](https://img.shields.io/pypi/v/owa-env-gst?label=owa-env-gst)](https://pypi.org/project/owa-env-gst/) | [![owa-env-gst](https://img.shields.io/conda/vn/conda-forge/owa-env-gst?label=conda)](https://anaconda.org/conda-forge/owa-env-gst) | [High-performance, hardware-accelerated](#high-performance-screen-capture) screen capture |
| [`owa.env.example`](https://github.com/open-world-agents/open-world-agents/tree/main/projects/owa-env-example) | - | - | Reference implementations for learning |

> 🎥 **Video Processing Packages**: Packages marked with 🎥 require GStreamer dependencies. Install `conda install open-world-agents::gstreamer-bundle` first for full functionality.

> 📦 **Lockstep Versioning**: All first-party OWA packages follow lockstep versioning, meaning they share the same version number to ensure compatibility and simplify dependency management.

> 💡 **Extensible Design**: Built for the community! Easily create custom plugins like `owa-env-minecraft` or `owa-env-web` to extend functionality.

## Installation

See the [Installation Guide](install.md) for detailed instructions.

## Desktop Recording with `ocap`

[**ocap**](https://github.com/open-world-agents/ocap) (Omnimodal CAPture) is a high-performance desktop recorder that captures screen video, audio, keyboard/mouse events, and window events in synchronized formats. Built with Windows APIs and GStreamer for hardware-accelerated recording with H265/HEVC encoding.

- **Complete recording**: Video + audio + keyboard/mouse + window events
- **High performance**: Hardware-accelerated, ~100MB/min for 1080p
- **Simple usage**: `ocap my-recording` (stop with Ctrl+C)
- **Modern formats**: [OWAMcap](data/getting-started/why-owamcap.md) with flexible MediaRef system (supports MKV, images, URLs, embedded data)

> 📖 **Detailed Documentation**: See [Desktop Recording Guide](data/getting-started/recording-data.md) for complete setup, usage examples, and troubleshooting.

## Environment Usage

<details>
<summary><strong>Environment Usage: Three Types of Components</strong></summary>

OWA's Environment provides three types of components for real-time agent interaction:

### Callables

Direct function calls for immediate actions:
```python
from owa.core import CALLABLES
# Components automatically available - zero configuration!

# Get current time, capture screen, click mouse
current_time = CALLABLES["std/time_ns"]()
screen = CALLABLES["desktop/screen.capture"]()
CALLABLES["desktop/mouse.click"]("left", 2)  # Double-click
```

### Listeners

Event monitoring with user-defined callbacks:
```python
from owa.core import LISTENERS
import time

# Monitor keyboard events
def on_key(event):
    print(f"Key pressed: {event.vk}")

listener = LISTENERS["desktop/keyboard"]().configure(callback=on_key)
with listener.session:
    input("Press Enter to stop...")

# Periodic tasks
def on_tick():
    print(f"Tick: {CALLABLES['std/time_ns']()}")

with LISTENERS["std/tick"]().configure(callback=on_tick, interval=1).session:
    time.sleep(3)  # Prints every second for 3 seconds
```

### Runnables

Background processes that can be started/stopped:
```python
from owa.core import RUNNABLES

# Periodic screen capture
capture = RUNNABLES["gst/screen_capture"]().configure(fps=60)
with capture.session:
    frame = capture.grab()
```

### Message Types

Access structured message definitions:
```python
from owa.core import MESSAGES

# Message types automatically available
KeyboardEvent = MESSAGES["desktop/KeyboardEvent"]
ScreenCaptured = MESSAGES["desktop/ScreenCaptured"]
```

</details>

## High-Performance Screen Capture

<!-- SYNC-ID: gst-performance-benchmark -->
Powered by GStreamer and Windows API, our implementation is **6x faster** than alternatives:

| **Library** | **Avg. Time per Frame** | **Relative Speed** |
|-------------|------------------------|--------------------|
| **owa.env.gst** | **5.7 ms** | ⚡ **1× (Fastest)** |
| `pyscreenshot` | 33 ms | 🚶‍♂️ 5.8× slower |
| `PIL` | 34 ms | 🚶‍♂️ 6.0× slower |
| `MSS` | 37 ms | 🚶‍♂️ 6.5× slower |
| `PyQt5` | 137 ms | 🐢 24× slower |

📌 **Tested on:** Intel i5-11400, GTX 1650
<!-- END-SYNC: gst-performance-benchmark -->

Not only does `owa.env.gst` **achieve higher FPS**, but it also maintains **lower CPU/GPU usage**, making it the ideal choice for screen recording. Same applies for `ocap`, since it internally imports `owa.env.gst`.

📊 **[See detailed benchmarks and methodology →](env/plugins/gst.md#performance)**

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

## OWAMcap Data Format

**OWAMcap** combines the robustness of [MCAP](https://mcap.dev/) with specialized desktop interaction schemas. Perfect synchronization of screen captures, input events, and window context with nanosecond precision.

<!-- SYNC-ID: owamcap-key-features -->
- 🌐 **Universal Standard**: Unlike fragmented formats, enables seamless dataset combination for large-scale foundation models *(OWAMcap)*
- ⚡ **High-Performance Multimodal Storage**: Lightweight [MCAP](https://mcap.dev/) container with nanosecond precision for synchronized data streams *(MCAP)*
- 🔗 **Flexible MediaRef**: Smart references to both external and embedded media (file paths, URLs, data URIs, video frames) with lazy loading - keeps metadata files small while supporting rich media *(OWAMcap)* → [Learn more](data/technical-reference/format-guide.md#media-handling)
- 🤗 **Training Pipeline Ready**: Native HuggingFace integration, seamless dataset loading, and direct compatibility with ML frameworks *(Ecosystem)* → [Browse datasets](https://huggingface.co/datasets?other=OWA) | [Data pipeline](data/technical-reference/data-pipeline.md)
<!-- END-SYNC: owamcap-key-features -->

> 📖 **Learn More**: [Why OWAMcap?](data/getting-started/why-owamcap.md) | [Complete Format Guide](data/technical-reference/format-guide.md) | [vs Other Formats](data/tools/comparison-with-lerobot.md)

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

## Desktop Recording & Dataset Sharing

Record your desktop usage data and share with the community:

```bash
# Install GStreamer dependencies (for video recording) and ocap
$ conda install open-world-agents::gstreamer-bundle && pip install ocap

# Record desktop activity (includes video, audio, events)
$ ocap my-session

# Upload to HuggingFace, browse community datasets!
# Visit: https://huggingface.co/datasets?other=OWA
```

## Community Datasets

<!-- SYNC-ID: community-datasets -->
**Browse Datasets**: [🤗 HuggingFace](https://huggingface.co/datasets?other=OWA)

- **Standardized Format**: All datasets use OWAMcap for seamless integration
- **Interactive Preview**: [Hugging Face Spaces Visualizer](https://huggingface.co/spaces/open-world-agents/visualize_dataset)
<!-- END-SYNC: community-datasets -->

**Access Community Datasets**:
```python
# Load datasets from HuggingFace
from owa.data import load_dataset

# Browse available OWAMcap datasets
datasets = load_dataset.list_available(format="OWA")

# Load a specific dataset. NOTE: Not implemented yet!
data = load_dataset("open-world-agents/D2E-480p")
```

## CLI Tools (`owl`)

```bash
# Data analysis
$ owl mcap info session.mcap              # File overview & statistics
$ owl mcap cat session.mcap --n 10        # View messages
$ owl video probe session.mkv             # Video analysis

# Environment management
$ owl env list                            # List plugins
$ owl env search "mouse.*click"           # Search components
$ owl messages show desktop/KeyboardEvent # View schemas
```

> **💡 Complete CLI Reference**: For detailed information about all CLI commands and options, see the [CLI Tools documentation](cli/index.md).

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

### Community Plugin Ideas

**Help us grow the ecosystem!** 🌱

| Example Name | Description |
|--------------|-------------|
| `owa.env.minecraft` | Minecraft automation & bot framework |
| `owa.env.web` | Browser automation via WebDriver |
| `owa.env.mobile` | Android/iOS device control |
| `owa.env.cad` | CAD software automation |
| `owa.env.trading` | Financial trading platform integration |

> 💡 **Want to contribute?** Check our [Plugin Development Guide](env/custom_plugins.md) to create your own `owa.env.*` package!
>
> 💭 **These are just examples!** The community decides what plugins to build. Propose your own ideas or create plugins for any domain you're passionate about.

## Documentation

- **Full Documentation**: https://open-world-agents.github.io/open-world-agents/
- **Environment Framework**: [Environment Guide](env/index.md) - Core concepts, usage guide, and plugin development
- **Data Infrastructure**: [Data Guide](data/index.md) - Recording, storage, and analysis with [OWAMcap format](data/getting-started/why-owamcap.md)
- **CLI Tools**: [CLI Reference](cli/index.md) - Command-line utilities and reference

## Contributing

We welcome contributions! Whether you're:
- Building new environment plugins
- Improving performance
- Adding documentation
- Reporting bugs

Please see our [Contributing Guide](contributing.md) for details.

## License

This project is released under the MIT License. See the [LICENSE](https://github.com/open-world-agents/open-world-agents/blob/main/LICENSE) file for details.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{choi2025d2e,
  title={D2E: Scaling Vision-Action Pretraining on Desktop Data for Transfer to Embodied AI},
  author={Choi, Suwhan and Jung, Jaeyoon and Seong, Haebin and Kim, Minchan and Kim, Minyeong and Cho, Yongjun and Kim, Yoonshik and Park, Yubeen and Yu, Youngjae and Lee, Yunsung},
  journal={arXiv preprint arXiv:2510.05684},
  year={2025}
}
```