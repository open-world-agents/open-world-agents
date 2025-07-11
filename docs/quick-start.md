# Quick Start Guide

**Complete step-by-step walkthrough for getting started with Open World Agents**

!!! tip "3-Step Workflow"
    This guide covers the complete OWA workflow: **Record** → **Process** → **Train**

## Overview

This guide provides detailed explanations, examples, and troubleshooting for the 3-step OWA workflow:

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

## Prerequisites

!!! warning "Installation Required"
    Before starting, ensure you have OWA installed. See the [Installation Guide](install.md) for detailed setup instructions.

=== "Video Recording"
    **For full recording capabilities:**
    ```bash
    # Install GStreamer dependencies first
    conda install open-world-agents::gstreamer-bundle
    pip install owa
    ```

=== "Data Processing Only"
    **For basic data processing:**
    ```bash
    pip install owa
    ```

## Step 1: Record Desktop Interaction

!!! example "Record with ocap"
    Use `ocap` (Omnimodal CAPture) to record your desktop interactions with synchronized video, audio, and input events.

    ```bash
    $ ocap my-session.mcap
    ```

!!! info "What this captures"
    - :material-video: **Screen video** with hardware acceleration
    - :material-keyboard: **Keyboard events** with nanosecond precision
    - :material-mouse: **Mouse interactions** with exact coordinates
    - :material-volume-high: **Audio recording** synchronized with video
    - :material-database: **Everything saved** in the [OWAMcap format](data/getting-started/why-owamcap.md)

!!! note "Learn More"
    - [Desktop Recording Guide](data/getting-started/recording-data.md) - Complete setup and usage
    - [OWAMcap Format](data/technical-reference/format-guide.md) - Technical specification
    - [Recording Troubleshooting](faq_dev.md) - Common issues and solutions

## Step 2: Process to Training Format

!!! example "Transform with Data Pipeline"
    Transform your recorded data into training-ready datasets using OWA's data pipeline.

    ```bash
    $ python scripts/01_raw_events_to_event_dataset.py --train-dir ./
    ```

!!! success "Processing Pipeline"
    - :material-export: **Extracts events** from the MCAP file
    - :material-cog: **Converts format** to standardized training structure
    - :material-link: **Handles media references** and synchronization
    - :material-brain: **Prepares data** for ML frameworks

!!! tip "Advanced Processing"
    ```mermaid
    flowchart LR
        A[MCAP File] --> B[Event Dataset]
        B --> C[Binned Dataset]
        C --> D[Training Ready]

        style A fill:#e1f5fe
        style D fill:#e8f5e8
    ```

!!! note "Learn More"
    - [Data Pipeline Guide](data/technical-reference/data-pipeline.md) - Complete processing workflow
    - [Data Explorer](data/getting-started/exploring-data.md) - Analyze and visualize your data
    - [CLI Tools](cli/index.md) - Command-line utilities for data management

## Step 3: Train Your Model

!!! warning "TODO: Training Implementation"
    This section is under development. Training scripts and detailed examples are coming soon.

!!! example "Train with Processed Data"
    Use the processed dataset to train your desktop agent model.

    ```bash
    $ python train.py --dataset ./event-dataset
    ```

!!! rocket "Training Capabilities"
    - :material-robot: **Multimodal models** on desktop interactions
    - :material-school: **Learn from demonstrations** - human behavior patterns
    - :material-application: **Application-specific agents** - tailored for your use case
    - :material-chart-line: **Performance evaluation** on real tasks

!!! abstract "Training Architecture"
    ```mermaid
    flowchart TD
        A[Event Dataset] --> B[Vision Encoder]
        A --> C[Action Encoder]
        B --> D[Multimodal Fusion]
        C --> D
        D --> E[Policy Network]
        E --> F[Desktop Agent]

        style A fill:#e1f5fe
        style F fill:#e8f5e8
    ```

!!! note "Learn More"
    - [Agent Examples](examples/) - Complete implementations and training pipelines
    - [Multimodal Game Agent](examples/multimodal_game_agent.md) - Vision-based game playing
    - [GUI Agent](examples/gui_agent.md) - General desktop automation
    - [Usage with LLMs](examples/usage_with_llm.md) - Integration patterns

## Environment Framework Integration

!!! tip "Real-time Agent Interactions"
    While recording and training, you can also use OWA's real-time environment framework for live agent interactions:

=== "Screen Capture"
    ```python
    from owa.core import CALLABLES

    # Real-time screen capture
    screen = CALLABLES["desktop/screen.capture"]()
    ```

=== "Event Monitoring"
    ```python
    from owa.core import LISTENERS

    # Monitor user interactions
    def on_key(event):
        print(f"Key pressed: {event.vk}")

    listener = LISTENERS["desktop/keyboard"]().configure(callback=on_key)
    ```

=== "Agent Actions"
    ```python
    from owa.core import CALLABLES

    # Perform desktop actions
    CALLABLES["desktop/mouse.click"]("left", 2)  # Double-click
    CALLABLES["desktop/keyboard.type"]("Hello World!")
    ```

!!! note "Learn More"
    - [Environment Guide](env/guide.md) - Complete system overview
    - [Environment Framework](env/index.md) - Core concepts and quick start
    - [Custom Plugins](env/custom_plugins.md) - Extend functionality

## Community Resources

!!! example "Datasets & Tools"
    === "Community Datasets"
        - :material-database: [Browse Community Datasets](https://huggingface.co/datasets?other=OWA) - Hundreds of OWAMcap datasets
        - :material-eye: [Dataset Visualizer](https://huggingface.co/spaces/open-world-agents/visualize_dataset) - Interactive preview tool

    === "Getting Help"
        - :material-help-circle: [FAQ](faq_dev.md) - Common questions and troubleshooting
        - :material-account-group: [Contributing Guide](contributing.md) - Development setup and contribution guidelines
        - :material-lifebuoy: [Help with OWA](help_with_owa.md) - Community support resources

## Next Steps

!!! success "Your Journey Continues"

    1. :material-code-braces: **Explore Examples**: Start with [Agent Examples](examples/) to see complete implementations

    2. :material-account-group: **Join the Community**: Browse and contribute [datasets](https://huggingface.co/datasets?other=OWA)

    3. :material-puzzle: **Build Custom Plugins**: Extend OWA with [custom environment plugins](env/custom_plugins.md)

    4. :material-book-open: **Advanced Usage**: Dive into [technical documentation](data/technical-reference/format-guide.md) for advanced features

!!! tip "Quick Links"
    - **Need help?** → [FAQ](faq_dev.md) or [Community Support](help_with_owa.md)
    - **Ready to build?** → [Agent Examples](examples/)
    - **Want to contribute?** → [Contributing Guide](contributing.md)
