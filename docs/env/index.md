# Environment Framework

**OWA's Env is the "USB-C of desktop agents"** - a universal interface for native desktop automation.

!!! info "Think MCP for Desktop"
    <!-- SYNC-ID: usb-c-analogy -->
    Just as [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) provides a standardized way for LLMs to connect to data sources and tools, **OWA's Env provides a standardized way for agents to connect to desktop environments**.

    - **MCP**: "USB-C of LLMs" - universal interface for AI tools
    - **OWA's Env**: "USB-C of desktop agents" - universal interface for native desktop automation
    <!-- END-SYNC: usb-c-analogy -->

!!! tip "Quick Start"
    ```bash
    pip install owa
    ```
    ```python
    from owa.core import CALLABLES, LISTENERS
    # Components automatically available - no configuration needed!
    ```

## Core Concepts

OWA's Environment provides three types of components for real-time agent interaction:

=== "Callables"
    **Direct function calls** for immediate actions
    ```python
    # Get current time, capture screen, click mouse
    CALLABLES["std/time_ns"]()
    CALLABLES["desktop/screen.capture"]()
    CALLABLES["desktop/mouse.click"]("left", 2)
    ```

=== "Listeners"
    **Event monitoring** with user-defined callbacks
    ```python
    # Monitor keyboard events
    def on_key(event):
        print(f"Key pressed: {event.vk}")

    listener = LISTENERS["desktop/keyboard"]().configure(callback=on_key)
    with listener.session:
        input("Press Enter to stop...")
    ```

=== "Runnables"
    **Background processes** that can be started/stopped
    ```python
    # Periodic screen capture
    capture = RUNNABLES["gst/screen_capture"]().configure(fps=60)
    with capture.session:
        frame = capture.grab()
    ```

## Why Choose OWA's Env?

Traditional frameworks like [gymnasium.Env](https://gymnasium.farama.org/api/env/) use synchronous `env.step()` calls that assume infinite processing time. **Real-world agents need real-time responses.**

<!-- SYNC-ID: env-framework-features -->
**OWA's Env delivers:**

- **⚡ Real-time Performance**: Optimized for responsive agent interactions (GStreamer components achieve <30ms latency)
- **🔌 Zero-Configuration**: Automatic plugin discovery via Python Entry Points
- **🌐 Event-Driven**: Asynchronous processing that mirrors real-world dynamics
- **🧩 Extensible**: Community-driven plugin ecosystem
<!-- END-SYNC: env-framework-features -->

## Quick Navigation

| Section | Description |
|---------|-------------|
| **[Environment Guide](guide.md)** | Complete system overview and usage examples |
| **[Custom Plugins](custom_plugins.md)** | Create your own environment extensions |
| **[CLI Tools](../cli/env.md)** | Plugin management and exploration commands |

**Built-in Plugins:**

| Plugin | Description | Key Features |
|--------|-------------|--------------|
| **[Standard](plugins/std.md)** | Core utilities | Time functions, periodic tasks |
| **[Desktop](plugins/desktop.md)** | Desktop automation | Mouse/keyboard control, window management |
| **[GStreamer](plugins/gst.md)** | High-performance capture | 6x faster screen recording |