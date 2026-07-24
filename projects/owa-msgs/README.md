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

### Discovering Available Message Types

To see all available message types, use the CLI command:

```bash
owl messages list
```

Or programmatically:

```python
from owa.core import MESSAGES

# List all available message types
for message_type in sorted(MESSAGES.keys()):
    print(f"- {message_type}")

# Get detailed information about a specific message
print(f"Total messages: {len(MESSAGES)}")
```

### Getting Message Details

For detailed information about any message type:

```bash
# Show comprehensive details about a message
owl messages show desktop/KeyboardEvent

# Show with usage examples
owl messages show desktop/KeyboardEvent --example

# Validate all messages
owl messages validate
```

## Message Domains

Messages are organized by domain for better structure:

- **Desktop Domain** (`desktop/*`): Desktop interaction messages including keyboard, mouse, window, and screen capture events

## Extending with Custom Messages

To add custom message types, create a package with entry point registration:

```toml
# pyproject.toml
[project.entry-points."owa.msgs"]
"custom/MyMessage" = "my_package.messages:MyMessage"
```

## Schema and Compatibility

All messages implement the `BaseMessage` interface from `owa.core.message` and are compatible with the OWAMcap format for data recording and playback.
