# owa-msgs

Core message definitions for Open World Agents (OWA).

This package provides the standard message types used throughout the OWA ecosystem, organized by domain and registered through Python entry points for automatic discovery.

## Installation

```bash
pip install owa-msgs
```

## Usage

### Using the Message Registry

The recommended way to access messages is through the global registry:

```python
from owa.core import MESSAGES

# Access message classes by type name
KeyboardEvent = MESSAGES['desktop/KeyboardEvent']
MouseEvent = MESSAGES['desktop/MouseEvent']

# Create message instances
event = KeyboardEvent(event_type="press", vk=65)
```

### Direct Imports

You can also import message classes directly:

```python
from owa.msgs.desktop.keyboard import KeyboardEvent
from owa.msgs.desktop.mouse import MouseEvent
```

### Available Message Types

- `desktop/KeyboardEvent` - Keyboard press/release events
- `desktop/KeyboardState` - Current keyboard state
- `desktop/MouseEvent` - Mouse movement, click, and scroll events  
- `desktop/MouseState` - Current mouse state

## Message Domains

### Desktop Domain

Messages related to desktop interaction:

- **Keyboard**: Key press/release events and state
- **Mouse**: Mouse movement, clicks, scrolls, and state

## Extending with Custom Messages

To add custom message types, create a package with entry point registration:

```toml
# pyproject.toml
[project.entry-points."owa.messages"]
"custom/MyMessage" = "my_package.messages:MyMessage"
```

## Schema and Compatibility

All messages implement the `BaseMessage` interface from `owa.core.message` and are compatible with the OWAMcap format for data recording and playback.
