# OWA Environment Example Plugin

This is an example environment plugin for Open World Agents that demonstrates the **OEP-0003** entry points-based plugin discovery system.

## Features

This plugin provides example implementations of all three component types:

- **Callables**: Functions and classes that can be called
  - `example/add`: Add two numbers
  - `example/print`: Print a message with formatting
  - `example/callable`: A callable class example

- **Listeners**: Event-driven components
  - `example/listener`: Periodic event generator
  - `example/timer`: One-shot timer

- **Runnables**: Background task processors
  - `example/runnable`: File-writing background task
  - `example/counter`: Counting task with auto-stop

## Installation

```bash
# Install in development mode
pip install -e .

# Or install from source
pip install .
```

## Usage

With OEP-0003, no manual activation is needed! Components are automatically discovered and registered when you import `owa.core`:

```python
from owa.core import CALLABLES, LISTENERS, RUNNABLES

# Components are automatically available
result = CALLABLES["example/add"](5, 3)
print(f"5 + 3 = {result}")

# Use the enhanced API
from owa.core import get_component

add_func = get_component("callables", namespace="example", name="add")
result = add_func(10, 20)
```

## Plugin Structure (OEP-0003)

This plugin follows the OEP-0003 standard:

1. **Entry Point Declaration** in `pyproject.toml`:
   ```toml
   [project.entry-points."owa.env.plugins"]
   example = "owa.env.plugins.example:plugin_spec"
   ```

2. **Plugin Specification** in `owa/env/plugins/example.py`:
   ```python
   """
   Plugin specification for the Example environment plugin.

   This module is kept separate to avoid circular imports during plugin discovery.
   """

   from owa.core.plugin_spec import PluginSpec

   plugin_spec = PluginSpec(
       namespace="example",
       version="0.1.0",
       description="Example plugin demonstrating the plugin system",
       author="OWA Development Team",
       components={
           "callables": {
               "add": "owa.env.example.example_callable:example_add",
               # ...
           },
           # ...
       }
   )
   ```

3. **Component Implementation**: Clean classes and functions without decorators

## Testing

Run the test script to verify the OEP-0003 implementation:

```bash
python test_oep_0003.py
```

## CLI Tools for Plugin Exploration

After installation, explore the plugin using the `owl env` CLI:

```bash
# List all plugins (including this example)
$ owl env list

# Show detailed information about the example plugin
$ owl env show example --components

# Search for specific components
$ owl env search "add" --namespace example

# Inspect a specific component
$ owl env show example --inspect add

# Quick exploration
$ owl env ls example                              # Quick overview
$ owl env find example                            # Search for example components
```