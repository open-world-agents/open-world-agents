# Standard Environment Plugin

The Standard Environment plugin (`owa.env.std`) is a core component of the Open World Agents framework. It provides essential functionalities related to time management and clock operations, which are fundamental for various time-based tasks and event scheduling within the system.

## Features

- **Time Functions**: The plugin registers functions like `std/time_ns` that return the current time in nanoseconds.
- **Tick Listener**: It includes a `std/tick` listener that can be configured to execute callbacks at specified intervals.

## Usage

The Standard Environment plugin is automatically available when you install `owa-core`. No manual activation needed!

```python
# Components automatically available after installation
from owa.core.registry import CALLABLES, LISTENERS
```

You can access the registered functions and listeners via the global registries using the unified `namespace/name` pattern:

```python
from owa.core.registry import CALLABLES, LISTENERS

# Get the current time in nanoseconds
current_time_ns = CALLABLES["std/time_ns"]()
print(f"Current time (ns): {current_time_ns}")

# Configure and start a tick listener
def on_tick():
    print(f"Tick at {CALLABLES['std/time_ns']()}")

tick_listener = LISTENERS["std/tick"]()
tick_listener.configure(callback=on_tick, interval=1)  # Tick every second
tick_listener.start()

# Run for a few seconds to see the tick listener in action
import time
time.sleep(5)

# Stop the tick listener
tick_listener.stop()
tick_listener.join()
```

## Components

### Time Functions

- **`std/time_ns`**: Returns the current time in nanoseconds. This function is registered in the `CALLABLES` registry.

### Tick Listener

- **`std/tick`**: A listener that triggers a callback at specified intervals. This listener is registered in the `LISTENERS` registry and can be configured with an interval in seconds.

## Example

Here is a complete example demonstrating how to use the Standard Environment plugin:

```python
from owa.core.registry import CALLABLES, LISTENERS

# Components automatically available - no activation needed!

# Print the current time in nanoseconds
print(CALLABLES["std/time_ns"]())

# Define a callback function for the tick listener
def tick_callback():
    print(f"Tick at {CALLABLES['std/time_ns']()}")

# Configure and start the tick listener
tick_listener = LISTENERS["std/tick"]().configure(callback=tick_callback, interval=1)
tick_listener.start()

# Let the listener run for 5 seconds
import time
time.sleep(5)

# Stop the tick listener
tick_listener.stop()
tick_listener.join()
```

This example demonstrates how to use the automatically available components, retrieve the current time, and set up a tick listener that prints the current time every second.

The Standard Environment plugin is a fundamental part of the Open World Agents framework, providing essential time-based functionalities that can be leveraged by other modules and applications.

## Auto-generated documentation

::: std
    handler: owa