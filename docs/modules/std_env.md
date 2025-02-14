# Standard Environment

The Standard Environment module (owa.env.std) is a core component of Open World Agents. It provides basic functionalities, such as time management and clock operations, which are essential for system operations.

## Features

- **Time Functions:** Registers functions like `clock.time_ns` that return the current time in nanoseconds.
- **Event Listener:** Activates event listeners (e.g., `clock/tick`) to periodically execute time-based callbacks.

## Usage

To activate the module, use the following command:

```python
activate_module("owa.env.std")
```

Once activated, functionalities can be accessed via the global `CALLABLES` and `LISTENERS` registries.
