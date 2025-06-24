# Open-sourcing Dataset for Multimodal Desktop Agent

## What is OWAMcap?

**OWAMcap** (Open World Agents MCAP) is a specialized file format for storing multimodal desktop interaction data. Think of it as a "video file" but for AI training data - it captures not just what you see on screen, but also every keyboard press, mouse movement, window change, and other desktop events, all perfectly synchronized with timestamps.

**Key Features:**
- 📹 **Screen recordings** with precise frame timestamps
- ⌨️ **Keyboard events** (every key press and release)
- 🖱️ **Mouse interactions** (movements, clicks, scrolls)
- 🪟 **Window information** (active windows, titles, positions)
- 🔗 **Perfect synchronization** - all events aligned to nanosecond precision
- 📦 **Self-contained** - everything needed for training in one file format

**What makes it special?** Unlike traditional screen recordings that only capture video, OWAMcap captures the complete "digital DNA" of human-computer interaction, making it perfect for training AI agents that need to understand and replicate human desktop behavior.

**Example:** When you click a button in an application, OWAMcap records:
```
Timestamp: 1234567890.123456789
Screen: [1920x1080 frame showing the button]
Mouse: click(x=450, y=300, button="left")
Window: title="My Application", position=(100, 50, 800, 600)
```
This rich context allows AI models to understand not just what happened visually, but the precise interaction that caused it.

## Why OWAMcap Matters

As of now (March 22, 2025), there are few datasets available for building multimodal desktop agents.

Even more scarce are datasets that (1) contain high-frequency screen data, (2) have keyboard/mouse information timestamp-aligned with other modalities like screen recordings, and (3) include human demonstrations.

**The Problem:** Most existing approaches either:
- Record only video (missing interaction context)
- Capture events separately (synchronization nightmares)
- Use proprietary formats (can't combine datasets)
- Lack precision (poor timestamps, low frequency)

**The Solution:** OWAMcap provides a unified, open standard that captures everything needed for desktop AI training in a single, efficient format.

## Our Complete Solution

To address this gap, open-world-agents provides the following four solutions:

1. **File Format - `OWAMcap`**: A high-performance, self-contained, flexible container file format for multimodal desktop log data, powered by the open-source container file format [mcap](https://mcap.dev/). [Learn more...](data_format.md)

2. **Message Definitions - `owa-msgs`**: A dedicated package providing core message definitions with automatic discovery through Python entry points. All message types are automatically available through the global `MESSAGES` registry from `owa.core`. [Learn more...](../env/guide.md#message-registry)

3. **Desktop Recorder - `ocap your-filename.mcap`**: A powerful, efficient, and easy-to-use desktop recorder that captures keyboard/mouse and high-frequency screen data.
    - Powered by [`owa-env-gst`](../env/plugins/gst.md), ensuring superior performance compared to alternatives. [Learn more...](ocap.md)

4. **🤗 [Hugging Face](https://huggingface.co/) Integration & Community Ecosystem**: The largest collection of open-source desktop interaction datasets in OWAMcap format.
    - **Growing Dataset Collection**: Hundreds of community-contributed datasets covering diverse workflows, applications, and interaction patterns
    - **Easy Upload & Sharing**: Upload your `ocap` recordings directly to HuggingFace with one command
    - **Standardized Format**: All datasets use the unified OWAMcap format for seamless integration
    - **Interactive Visualization**: Preview any dataset at [Hugging Face Spaces](https://huggingface.co/spaces/open-world-agents/visualize_dataset)
    - **Browse Available Datasets**: [🤗 datasets?other=OWA](https://huggingface.co/datasets?other=OWA)

> 🚀 **Community Impact**: With OWA's streamlined recording and sharing pipeline, the open-source desktop agent community has rapidly grown from zero to hundreds of publicly available multimodal datasets, democratizing access to high-quality training data.