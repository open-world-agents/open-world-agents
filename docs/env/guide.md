# Comprehensive Guide for Env

## Core Concepts

### Three Main Components:

Open World Agents (OWA)'s **Env** consists of three primary components that enable interaction with the environment in different ways.

1. **Callable** - Functions you actively call to perform actions or get state
    - *These are like traditional function calls; you invoke them when you need to perform an action or retrieve some information from the environment.*
    - Implements `__call__` function
    - Example: `CALLABLES["std/time_ns"]()`

2. **Listener** - Components that respond to events and execute your callbacks
    - *Listeners wait for specific events and execute your callback functions when those events occur.*
    - Takes a `callback` parameter in the `configure` method
    - Example:
```python
listener = LISTENERS["keyboard"]().configure(callback=my_callback)
with listener.session:
    input("Type enter to exit.")
```
*This example sets up a keyboard listener that invokes `my_callback` whenever a keyboard event is detected.*

3. **Runnable** - Background processes that can be started and stopped
    - *Runnables run in the background and can be managed with start and stop operations.*
    - Parent class of `Listener`, the only difference is absence of `callback` argument in `configure`.
    - Supports `start()`, `stop()`, and `join()` operations

!!! callable vs listener "What's the difference between **Callable** and **Listener**?"
    The key difference between these two is who initiates the call:

    - In **Callable**, **caller** actively executes the Callable.
    - In **Listener**, **callee** waits for events and then calls user-provided "callbacks".

    *In other words, Callables are synchronous functions you call directly, while Listeners are asynchronous and react to events.*

    Common environmental interfaces such as [gymnasium.Env](https://gymnasium.farama.org/api/env/) only provides object/method equivalent to **Callable**.

### Registry System

*The OWA environment uses a registry system to manage and access the various components.*

Components are managed through global registries:

- `CALLABLES` - Dictionary of callable functions

- `LISTENERS` - Dictionary of event listeners

- `RUNNABLES` - Dictionary of background processes

**Zero-Configuration Plugin Discovery**: Plugins are automatically discovered when installed via pip using Python's Entry Points system. No manual activation needed!

```python
from owa.core.registry import CALLABLES, LISTENERS, RUNNABLES
# Components automatically available after plugin installation
```

*All components use unified `namespace/name` naming pattern for consistency.*

## Environment Usage Examples

### Standard Environment (`owa.env.std`)

*Here is an example of how to use the standard environment to interact with clock functionalities.*

```python
import time
from owa.core.registry import CALLABLES, LISTENERS

# Components automatically available - no activation needed!
# Unified namespace/name pattern: std/time_ns, std/tick

# Testing the std/tick listener
tick = LISTENERS["std/tick"]().configure(callback=lambda: print(CALLABLES["std/time_ns"]()), interval=1)
tick.start()

time.sleep(2)  # The listener prints the current time in nanoseconds a few times

tick.stop(), tick.join()
```
*Components are automatically discovered and available after installation. All components use the unified `namespace/name` pattern.*


Instead of manual `start-stop-join` procedure, you may utilize context manager: `.session`! Following example shows how to abbreviate `start-stop-join` steps.

```python
with tick.session:
    time.sleep(2)
```

### Desktop Environment (`owa.env.desktop`)

*The desktop environment module provides capabilities for UI interaction and input handling.*

```python
from owa.core.registry import CALLABLES, LISTENERS
from owa.msgs.desktop.keyboard import KeyboardEvent

# Components automatically available - unified namespace/name pattern

# Using screen capture and window management features
print(f"{CALLABLES['desktop/screen.capture']().shape=}")  # Example output: (1080, 1920, 3)
print(f"{CALLABLES['desktop/window.get_active_window']()=}")
print(f"{CALLABLES['desktop/window.get_window_by_title']('open-world-agents')=}")

# Simulating a mouse click (left button, double click)
mouse_click = CALLABLES["desktop/mouse.click"]
mouse_click("left", 2)


# Configuring a keyboard listener
def on_keyboard_event(keyboard_event: KeyboardEvent):
    print(f"Keyboard event: {keyboard_event.event_type=}, {keyboard_event.vk=}")


keyboard_listener = LISTENERS["desktop/keyboard"]().configure(callback=on_keyboard_event)
with keyboard_listener.session:
    input("Type enter to exit.\n")
```
*Components are automatically available with unified naming. This code demonstrates capturing the screen, retrieving window information, simulating mouse clicks, and listening to keyboard events.*

### Custom EnvPlugin Example

You can create your own plugins using Entry Points for automatic discovery. For more information, see [Custom EnvPlugin](custom_plugins.md).

*Creating custom plugins allows you to extend the OWA environment with your own functionalities.*

```python
# In your plugin's pyproject.toml:
# [project.entry-points."owa.env.plugins"]
# myplugin = "owa.env.myplugin:plugin_spec"

# In your plugin specification:
from owa.core.plugin_spec import PluginSpec

plugin_spec = PluginSpec(
    namespace="myplugin",
    version="0.1.0",
    description="My custom plugin",
    components={
        "callables": {
            "add": "owa.env.myplugin:add_function",
        },
        "listeners": {
            "events": "owa.env.myplugin:EventListener",
        }
    }
)

# Using the custom plugin (automatically available after pip install)
from owa.core.registry import CALLABLES, LISTENERS
result = CALLABLES["myplugin/add"](5, 3)  # Returns 8
```
*Plugins use Entry Points for automatic discovery and unified `namespace/name` pattern for all components.*

## Architecture Summary

*The diagram below summarizes the architecture of the OWA environment and how components are registered and used.*

```mermaid
graph LR;
    EP[Entry Points] -->|Auto-discovers| SM["Standard Plugin(owa.env.std)"]
    EP -->|Auto-discovers| DM["Desktop Plugin(owa.env.desktop)"]
    EP -->|Auto-discovers| MP["Message Package(owa-msgs)"]
    SM -->|Provides| C1[std/time_ns]
    SM -->|Provides| L1[std/tick Listener]
    DM -->|Provides| C2[desktop/screen.capture]
    DM -->|Provides| C3[desktop/window.get_active_window]
    DM -->|Provides| L2[desktop/keyboard Listener]
    MP -->|Provides| M1[desktop/KeyboardEvent]
    MP -->|Provides| M2[desktop/MouseEvent]
    MP -->|Provides| M3[desktop/ScreenCaptured]
    User -->|pip install| PI[Plugin Installation]
    PI --> EP
    EP --> R[Component Registry]
    EP --> MR[Message Registry]
```

## CLI Tools for Plugin Management

The `owl env` command provides powerful tools for managing and exploring plugins with enhanced filtering, search, and analysis capabilities:

### Plugin Discovery and Listing

```bash
# List all discovered plugins
$ owl env list

# Enhanced filtering and display options
$ owl env list --namespace example                # Filter by namespace
$ owl env list --type callables                   # Filter by component type
$ owl env list --search mouse                     # Search by pattern
$ owl env list --details                          # Show import paths and load status
$ owl env list --table                            # Display in table format
$ owl env list --sort name                        # Sort by name, namespace, or type
```

### Plugin Information and Inspection

```bash
# Show plugin summary
$ owl env show example

# Enhanced plugin exploration
$ owl env show example --components               # Show detailed component list
$ owl env show example --inspect mouse.click      # Inspect specific component
$ owl env show example --type callables           # Filter by component type
$ owl env show example --search window            # Search within namespace
$ owl env show example --table                    # Display in table format
```

### Advanced Search and Discovery

```bash
# Search across all plugins
$ owl env search "mouse.*click"                   # Regex pattern search
$ owl env search window --details --table         # Detailed search results
$ owl env search keyboard --type callables        # Search specific component type

# Quick access shortcuts
$ owl env ls                                       # Quick plugin list
$ owl env ls desktop                              # Quick namespace exploration
$ owl env find mouse                              # Quick component search
$ owl env namespaces                              # List all available namespaces
```

### Ecosystem Analysis

```bash
# Statistics and health monitoring
$ owl env stats                                    # Show ecosystem statistics
$ owl env stats --by-namespace                    # Group by namespace
$ owl env stats --by-type                         # Group by component type
$ owl env health                                   # Perform health check
```

### Plugin Development

```bash
# Validate plugin specifications
$ owl env validate owa.env.myplugin:plugin_spec    # Python entry point
$ owl env validate ./plugin.yaml                   # YAML file
$ owl env validate ./plugin.yaml --verbose         # Detailed validation
$ owl env validate ./plugin.yaml --no-check-imports # Skip import validation
```

### Example CLI Output

```bash
$ owl env list
📦 Discovered Plugins (4)
├── desktop (25 components)
│   ├── 📞 Callables: 20
│   └── 👂 Listeners: 5
├── example (7 components)
│   ├── 📞 Callables: 3
│   ├── 👂 Listeners: 2
│   └── 🏃 Runnables: 2
├── gst (4 components)
│   ├── 👂 Listeners: 2
│   └── 🏃 Runnables: 2
└── std (2 components)
    ├── 📞 Callables: 1
    └── 👂 Listeners: 1

$ owl env show example --components
📦 Plugin: example (7 components)
├── 📞 Callables: 3
├── 👂 Listeners: 2
└── 🏃 Runnables: 2
📞 Callables (3)
├── example/add
├── example/callable
└── example/print
👂 Listeners (2)
├── example/listener
└── example/timer
🏃 Runnables (2)
├── example/counter
└── example/runnable

$ owl env search "mouse" --table
               Search Results for 'mouse' (9 matches)
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Component               ┃ Type      ┃ Namespace ┃ Name            ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ desktop/mouse           │ listeners │ desktop   │ mouse           │
│ desktop/mouse.click     │ callables │ desktop   │ mouse.click     │
│ desktop/mouse.move      │ callables │ desktop   │ mouse.move      │
│ desktop/mouse.press     │ callables │ desktop   │ mouse.press     │
│ desktop/mouse.scroll    │ callables │ desktop   │ mouse.scroll    │
└─────────────────────────┴───────────┴───────────┴─────────────────┘

$ owl env namespaces
             Available Namespaces
┏━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Namespace ┃ Components ┃ Quick Access       ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ desktop   │        25 │ owl env ls desktop │
│ example   │         7 │ owl env ls example │
│ gst       │         4 │ owl env ls gst     │
│ std       │         2 │ owl env ls std     │
└───────────┴────────────┴────────────────────┘
```

## Message Registry

OWA provides a centralized message registry system that automatically discovers and manages message definitions through Python entry points. This system separates message schemas from runtime components, providing better organization and extensibility.

### Accessing Messages

```python
from owa.core import MESSAGES

# Access message classes by type name
KeyboardEvent = MESSAGES['desktop/KeyboardEvent']
MouseEvent = MESSAGES['desktop/MouseEvent']
ScreenCaptured = MESSAGES['desktop/ScreenCaptured']

# Check if a message type exists
if 'desktop/KeyboardEvent' in MESSAGES:
    print("KeyboardEvent is available")

# List all available message types
for message_type in MESSAGES.keys():
    print(f"Available: {message_type}")

# Create message instances
event = KeyboardEvent(event_type="press", vk=65, timestamp=1234567890)
```

### Message Naming Convention

Messages follow a domain-based naming pattern:
- **Format**: `domain/MessageType`
- **Domain**: Logical grouping (e.g., `desktop`, `sensors`, `system`)
- **MessageType**: PascalCase message name
- **Examples**: `desktop/KeyboardEvent`, `desktop/WindowInfo`, `sensors/TemperatureReading`

### Core Message Types

The `owa-msgs` package provides standard message definitions:

| Message Type | Description |
|--------------|-------------|
| `desktop/KeyboardEvent` | Keyboard press/release events |
| `desktop/KeyboardState` | Current keyboard state |
| `desktop/MouseEvent` | Mouse movement, clicks, scrolls |
| `desktop/MouseState` | Current mouse position and buttons |
| `desktop/ScreenCaptured` | Screen capture frames with timestamps |
| `desktop/WindowInfo` | Active window information |

### CLI Tools for Message Management

```bash
# List all available message types
$ owl messages list

# Show detailed message schema
$ owl messages show desktop/KeyboardEvent

# Search for specific message types
$ owl messages search keyboard

# Validate message definitions
$ owl messages validate
```

### Custom Message Registration

Third-party packages can register custom message types through entry points:

```toml
# pyproject.toml
[project.entry-points."owa.msgs"]
"sensors/TemperatureReading" = "custom_sensors.messages:TemperatureReading"
"sensors/HumidityReading" = "custom_sensors.messages:HumidityReading"
```

## Additional Resources

- For standard module details: [owa-env-std](plugins/std.md)
- For desktop features: [owa-env-desktop](plugins/desktop.md)
- For multimedia support: [owa-env-gst](plugins/gst.md)
- For custom EnvPlugin development: [custom_plugins.md](custom_plugins.md)