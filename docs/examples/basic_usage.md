# Basic Usage

This document provides several basic usage examples for Open World Agents to help you get started quickly.

## Example 1: Activating Modules

Activate the Standard Environment module to register time-based functionalities:

```python
from owa.registry import activate_module

activate_module("owa.env.std")
print(CALLABLES)  # Should now include clock.time_ns
```

## Example 2: Handling Events

Set up a simple clock/tick listener to periodically print the current time:

```python
# Assuming CALLABLES and LISTENERS are imported from owa.registry

tick = LISTENERS["clock/tick"]()
tick.configure(callback=lambda: print(CALLABLES["clock.time_ns"]()), interval=1)
tick.activate()

import time
time.sleep(2)

tick.deactivate()
tick.shutdown()
```

## Example 3: Interacting with the Desktop Environment

Activate the Desktop Environment module to capture the screen and manage windows:

```python
activate_module("owa.env.desktop")

# Capture screen dimensions
print(CALLABLES["screen.capture"]().shape)

# Retrieve active window
print(CALLABLES["window.get_active_window"])()
```

These examples illustrate the fundamental usage patterns of module activation, event management, and interactivity with desktop systems.
