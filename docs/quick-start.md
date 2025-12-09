# Quick Start Guide

This guide covers the OWA workflow: **Record** → **Process** → **Train**

!!! info "Training Pipeline Coming Soon"
    We developed a complete training pipeline during our [D2E research](https://worv-ai.github.io/d2e/). We're currently preparing it for open-source release—stay tuned!

<!-- SYNC-ID: quick-start-3-steps -->
```bash
# 1. Record desktop interaction
$ ocap my-session.mcap

# 2. Process to training format
$ python scripts/01_raw_to_event.py --train-dir ./

# 3. Train your model (coming soon)
$ python train.py --dataset ./event-dataset
```
<!-- END-SYNC: quick-start-3-steps -->

## Prerequisites

Before starting, install OWA. See the [Installation Guide](install.md) for details.

=== "With Video Recording"
    ```bash
    $ conda install open-world-agents::gstreamer-bundle
    $ pip install owa
    ```

=== "Data Processing Only"
    ```bash
    $ pip install owa
    ```

## Step 1: Record Desktop Interaction

[`ocap`](https://github.com/open-world-agents/ocap) records your desktop in one command:

```bash
$ ocap my-session.mcap
```

This captures screen video (H.265), keyboard/mouse events, window context, and audio—all synchronized with nanosecond precision. See [`ocap` documentation](https://github.com/open-world-agents/ocap) for options.

Here's a demo of `ocap` in action:

<video src="https://github.com/user-attachments/assets/4e94782c-02ae-4f64-bb52-b08be69d33da" controls width="100%"></video>

## Step 2: Process to Training Format

Transform recorded data into training-ready datasets:

![Data Pipeline](https://raw.githubusercontent.com/open-world-agents/open-world-agents/main/projects/owa-data/pipeline.svg)

See [owa-data](https://github.com/open-world-agents/open-world-agents/tree/main/projects/owa-data) for full pipeline documentation.

## Step 3: Train Your Model

!!! info "Training Pipeline Coming Soon"
    We developed a complete training pipeline during our [D2E research](https://worv-ai.github.io/d2e/). We're currently preparing it for open-source release—stay tuned!

## Environment Framework

For live agent interactions (not just recording), use OWA's environment framework:

=== "Screen Capture"
    ```python
    from owa.core import CALLABLES

    screen = CALLABLES["desktop/screen.capture"]()
    ```

=== "Mouse/Keyboard Control"
    ```python
    from owa.core import CALLABLES

    CALLABLES["desktop/mouse.click"]("left", 2)  # Double-click
    CALLABLES["desktop/keyboard.type"]("Hello World!")
    ```

=== "Event Monitoring"
    ```python
    from owa.core import LISTENERS

    def on_key(event):
        print(f"Key pressed: {event.vk}")

    listener = LISTENERS["desktop/keyboard"]().configure(callback=on_key)
    ```

See [Environment Guide](env/guide.md) for the full API.

## Next Steps

| Goal | Resource |
|------|----------|
| Browse community data | [🤗 HuggingFace Datasets](https://huggingface.co/datasets?other=OWA) |
| Visualize recordings | [Dataset Visualizer](https://huggingface.co/spaces/open-world-agents/visualize_dataset) |
| Build agents | [Agent Examples](examples/index.md) |
| Extend OWA | [Custom Plugins](env/custom_plugins.md) |
| Get help | [FAQ](faq_dev.md) · [Contributing](contributing.md)
