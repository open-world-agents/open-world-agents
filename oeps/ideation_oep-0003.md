# Entry Points-Based Plugin Discovery and Unified Naming for OWA

## Vision: Standard Python Plugin Ecosystem

Imagine a developer Bob who wants to create an environment plugin "example". Following Python's standard plugin architecture using Entry Points (as described in the [official Python packaging guide](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/)), Bob declares his plugin in `pyproject.toml`. The system automatically discovers and registers all installed plugins without requiring manual activation.

## Python Standards for Plugin Discovery

We follow the official Python packaging guide's **"Using package metadata"** approach with Entry Points. This is the most robust and standard method for plugin discovery in Python.

## Unified Naming Convention

All components now use the same `namespace/name` pattern:

- **Callables**: `example/print`, `example/add`, `std/time_ns`
- **Listeners**: `example/events`, `std/tick`
- **Runnables**: `example/processor`, `std/monitor`

This eliminates the inconsistency between `namespace.name` (callables) and `namespace/name` (listeners/runnables).

## Entry Points-Based Plugin Declaration

### Plugin Package Structure (Hybrid Approach)

Bob can choose between two structures based on plugin complexity:

**Option 1: Simple Module (for small plugins)**
```
owa-env-example/
├── pyproject.toml                    # Entry point declaration
├── README.md
├── owa/
│   └── env/
│       └── example.py                # All components in one file
└── tests/
    └── test_example.py
```

**Option 2: Package Structure (for complex plugins)**
```
owa-env-example/
├── pyproject.toml                    # Entry point declaration
├── README.md
├── owa/
│   └── env/
│       └── example/
│           ├── __init__.py           # Plugin specification
│           ├── callables.py          # Callable implementations
│           ├── listeners.py          # Listener implementations
│           └── runnables.py          # Runnable implementations
└── tests/
    ├── test_callables.py
    └── test_listeners.py
```

### Entry Point Declaration

```toml
# pyproject.toml
[project]
name = "owa-env-example"
version = "0.1.0"
description = "Example environment plugin for Open World Agents"
dependencies = ["owa-core"]

[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

# Entry point declaration - this is the key part!
[project.entry-points."owa.env.plugins"]
example = "owa.env.example:plugin_spec"
```

### Plugin Specification

```python
# owa/env/example.py (for simple module) or owa/env/example/__init__.py (for package)
from owa.core.plugin_spec import PluginSpec

plugin_spec = PluginSpec(
    namespace="example",
    version="0.1.0",
    description="Example environment plugin",
    author="Bob's AI Solutions",
    components={
        "callables": {
            "print": "owa.env.example:enhanced_print",  # Simple module
            "add": "owa.env.example:add_numbers",
        },
        "listeners": {
            "events": "owa.env.example:EventListener",
        },
        "runnables": {
            "processor": "owa.env.example:DataProcessor",
        }
    }
)

# Component implementations can be in the same file (simple) or imported from submodules (complex)
```

### Example Implementation

**Simple Module Approach (owa/env/example.py):**
```python
from owa.core.plugin_spec import PluginSpec

# Plugin specification
plugin_spec = PluginSpec(
    namespace="example",
    version="0.1.0",
    description="Example environment plugin",
    author="Bob's AI Solutions",
    components={
        "callables": {
            "add": "owa.env.example:add_numbers",
            "print": "owa.env.example:enhanced_print",
        },
        "listeners": {
            "events": "owa.env.example:EventListener",
        },
        "runnables": {
            "processor": "owa.env.example:DataProcessor",
        }
    }
)

# Component implementations in the same file
def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def enhanced_print(message: str, level: str = "info") -> str:
    """Enhanced printing with formatting."""
    formatted = f"[{level.upper()}] {message}"
    print(formatted)
    return formatted

class EventListener(Listener):
    """Example event listener."""

class DataProcessor(Runnable):
    """Example runnable component."""
    def on_configure(self, batch_size: int):
        ...
```

## Entry Points-Based Plugin Discovery

### Zero-Configuration Usage

Users install Bob's plugin using standard pip:

```bash
pip install owa-env-example
```

**No `activate_module()` needed!** The system automatically discovers plugins via Entry Points and registers all components:

```python
from owa.core.registry import CALLABLES, LISTENERS, RUNNABLES

# Components are automatically available after installation
# All use unified namespace/name pattern

# Use callables
result = CALLABLES["example/add"](5, 3)
print(f"5 + 3 = {result}")

# Use listeners
def on_event(data):
    print(f"Received: {data}")

listener = LISTENERS["example/events"]().configure(callback=on_event)
with listener.session:
    # Do something that triggers events
    pass

# Use runnables
processor = RUNNABLES["example/processor"]().configure(batch_size=100)
processor.start()
```

### Flexible Component Access with Lazy Loading

The system provides multiple ways to access components with built-in lazy loading for optimal performance:

```python
from owa.core.registry import CALLABLES, get_component, list_components

# Method 1: Direct registry access (unified naming) - import happens here
add_func = CALLABLES["example/add"]  # Module imported when component is retrieved from registry
result = add_func(5, 3)  # Function already loaded, just execute

# Method 2: Alternative syntax for better ergonomics
add_func = get_component("callables", "example", "add")  # Import happens during retrieval
result = add_func(5, 3)

# Method 3: Get all components in a namespace - imports happen during retrieval
example_callables = get_component("callables", namespace="example")
# Returns: {"add": <function>, "print": <function>} - all imported when retrieved

# Method 4: List available components (metadata only, no imports)
all_callables = list_components("callables")  # Only metadata, no imports
example_only = list_components("callables", namespace="example")
```

### Lazy Loading Requirements

**All plugin components must be lazy-loaded** to ensure optimal startup performance and memory usage:

1. **Registration Phase**: Only plugin metadata and import paths are stored, no actual imports occur
2. **Retrieval Phase**: Component modules are imported only when retrieved from registry (`CALLABLES["name"]`)
3. **Execution Phase**: Already-imported components are executed directly
4. **Caching**: Once imported during retrieval, components are cached for subsequent retrievals
5. **Error Handling**: Import errors occur during retrieval, allowing graceful handling and partial plugin functionality

### Interface Design

New design supports two primary access patterns:

```python
# Pattern 1: Dictionary-style access
result = CALLABLES["example/add"](5, 3)

# Pattern 2: Function-based access
result = get_component("callables", "example", "add")(5, 3)
```

## Entry Points Discovery Implementation

### Plugin Discovery and Registration

The system automatically discovers plugins via Entry Points on startup:

```python
import sys
if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points

discovered_plugins = entry_points(group='owa.env.plugins')
```

### How Entry Points Discovery Works with Lazy Loading

1. **Plugin Declaration**: Plugins declare themselves in `pyproject.toml`:
   ```toml
   [project.entry-points."owa.env.plugins"]
   example = "owa.env.example:plugin_spec"
   ```

2. **Automatic Discovery**: On OWA startup, the system scans all installed packages for entry points in the `"owa.env.plugins"` group

3. **Plugin Metadata Loading**: For each entry point found:
   - Load the specified module (`owa.env.example`)
   - Get the specified object (`plugin_spec`)
   - Validate it's a `PluginSpec` instance

4. **Component Registration (Lazy)**: Register component metadata only:
   - Store import paths (`"owa.env.example:add_numbers"`) in registry
   - **No actual imports occur** - components remain unloaded
   - Use unified `namespace/name` pattern for registration

5. **Component Retrieval (Lazy Loading)**: When `CALLABLES["example/add"]` is called:
   - Check if component is already loaded (cache hit)
   - If not loaded, import the module and get the component
   - Cache the imported component for future use
   - Return the loaded component

6. **Error Handling**: Import errors are deferred to retrieval time, allowing partial plugin functionality

## Enhanced Component Access API

### Unified Component Access Functions

```python
from owa.core.registry import CALLABLES, LISTENERS, RUNNABLES
from typing import Dict, List, Optional, Any

def get_component(component_type: str, namespace: str = None, name: str = None) -> Any:
    """
    Flexible component access with multiple patterns.

    Usage patterns:
    1. get_component("callables", namespace="example", name="add")
    2. get_component("callables", namespace="example")  # Returns all in namespace
    3. get_component("callables")  # Returns all callables
    """
    registry = get_registry(component_type)
    if not registry:
        raise ValueError(f"Unknown component type: {component_type}")

    if namespace and name:
        # Get specific component: namespace/name
        full_name = f"{namespace}/{name}"
        return registry.get(full_name)

    elif namespace:
        # Get all components in namespace
        components = {}
        for full_name in registry._registry:
            if full_name.startswith(f"{namespace}/"):
                component_name = full_name.split("/", 1)[1]
                components[component_name] = registry._registry[full_name]
        return components

    else:
        # Get all components
        return dict(registry._registry)

def list_components(component_type: str = None, namespace: str = None) -> Dict[str, List[str]]:
    """
    List available components with optional filtering.

    Returns:
        Dictionary mapping component types to lists of component names
    """
    if component_type:
        registries = {component_type: get_registry(component_type)}
    else:
        registries = {
            "callables": CALLABLES,
            "listeners": LISTENERS,
            "runnables": RUNNABLES,
        }

    result = {}
    for reg_type, registry in registries.items():
        if not registry:
            continue

        components = []
        for full_name in registry._registry:
            if namespace:
                # Filter by namespace
                if full_name.startswith(f"{namespace}/"):
                    components.append(full_name)
            else:
                components.append(full_name)

        result[reg_type] = components

    return result

def get_registry(component_type: str):
    """Get the appropriate registry for component type."""
    registries = {
        "callables": CALLABLES,
        "listeners": LISTENERS,
        "runnables": RUNNABLES,
    }
    return registries.get(component_type)

# Usage examples
# Get specific component
add_func = get_component("callables", namespace="example", name="add")
result = add_func(5, 3)

# Get all callables in example namespace
example_callables = get_component("callables", namespace="example")
# Returns: {"add": <function>, "print": <function>, "calculate": <function>}

# List all components in example namespace
example_components = list_components(namespace="example")
# Returns: {"callables": ["example/add", "example/print"], "listeners": ["example/events"]}

# List all callables across all namespaces
all_callables = list_components("callables")
# Returns: {"callables": ["example/add", "std/time_ns", "desktop/click"]}
```

## Enhanced CLI Tools

The `owl` command-line interface supports the new unified system:

### Plugin and Component Management Commands

```bash
# Plugin discovery and listing (automatic - no activation needed)
$ owl env list                             # List all discovered plugins
$ owl env list --namespace example         # List plugins in specific namespace

# Plugin information
$ owl env show example                      # Show plugin information
$ owl env show example --components        # Show plugin components

# Component listing (unified namespace/name format)
$ owl env list callables                   # List all callables
$ owl env list callables --namespace example  # List callables in namespace
$ owl env list listeners                   # List all listeners
$ owl env list runnables                   # List all runnables

# Plugin validation (for development)
$ owl env validate ./plugin.yaml          # Validate plugin specification
```

### CLI Output Examples (Unified Naming)

```bash
$ owl env list
📦 Discovered Plugins:
├── std (v0.1.0) - Standard environment functions
├── desktop (v0.2.0) - Desktop interaction capabilities
└── example (v0.1.0) - Example plugin by Bob

$ owl env show example
📦 Plugin: example (v0.1.0)
Author: Bob's AI Solutions
Description: Example environment plugin

🔧 Components (unified namespace/name):
├── Callables: example/print, example/add, example/calculate
├── Listeners: example/events, example/timer
└── Runnables: example/processor

$ owl env list callables --namespace example
📞 Callables in 'example':
├── example/print - Enhanced printing function
├── example/add - Add two numbers
└── example/calculate - Advanced calculator

$ owl env list callables
📞 All Callables:
├── std/time_ns - Get current time in nanoseconds
├── desktop/click - Simulate mouse click
├── example/print - Enhanced printing function
├── example/add - Add two numbers
└── example/calculate - Advanced calculator
```

## Implementation Roadmap

### Phase 1: Entry Points-Based Discovery with Lazy Loading (Next Steps)
- 🔄 Remove `activate_module()` requirement
- 🔄 Implement Entry Points-based plugin discovery on startup
- 🔄 **Implement lazy loading for all plugin components**
- 🔄 Unify all component naming to `namespace/name` format
- 🔄 Remove `@CALLABLES.register()` and similar decorators from all plugins
- 🔄 **Add enhanced component access API with lazy loading support**
- 🔄 **Implement low-level lazy loading control interface**
- 🔄 Update CLI tools to support new patterns

### Phase 2: Interface Enhancement and Developer Experience
- 📋 **Evaluate and implement improved component access interfaces**
- 📋 **Add type safety and IDE support for component access**
- 📋 Plugin template generator (`owl env create`)
- 📋 Plugin validation tools
- 📋 Better documentation and examples
- 📋 Migration tools for existing plugins
- 📋 **Performance monitoring and optimization tools for lazy loading**

### Key Design Changes

1. **Standard Plugin Discovery**: Use Python's standard Entry Points mechanism
2. **Zero Configuration**: No more `activate_module()` - plugins work immediately after `pip install`
3. **Unified Naming**: All components use `namespace/name` pattern consistently
4. **Lazy Loading**: Components are imported only when retrieved from registry, not during registration
5. **Enhanced Access Patterns**: Multiple interface options including `CALLABLES["a/b"]` and `get_component("callables", "a", "b")`
6. **Flexible Access**: Multiple ways to access components (direct, by namespace, by type)
7. **Robust Discovery**: Entry Points provide reliable, metadata-based plugin discovery

### Migration from Current System

**Before (Current OWA):**
```python
from owa.core.registry import CALLABLES, LISTENERS, activate_module

activate_module("owa.env.std")
activate_module("owa.env.example")

time_func = CALLABLES["clock.time_ns"]  # dot notation
listener = LISTENERS["example/events"]  # slash notation
```

**After (Entry Points System):**
```python
from owa.core.registry import CALLABLES, LISTENERS
# No activate_module needed!

time_func = CALLABLES["std/time_ns"]    # unified slash notation
listener = LISTENERS["example/events"]  # same slash notation

# Or use flexible access
time_func = get_component("callables", namespace="std", name="time_ns")
example_callables = get_component("callables", namespace="example")
```

### Benefits

1. **Python Standard**: Uses official Python packaging standards for plugin discovery
2. **Simpler Usage**: No need to remember which plugins to activate
3. **Consistent Naming**: All components follow the same pattern
4. **Optimal Performance**: Lazy loading ensures fast startup and minimal memory usage
5. **Better Discovery**: Easy to explore what's available without loading everything
6. **Flexible Access**: Multiple interface options and ways to get components based on use case
7. **Advanced Control**: Low-level lazy loading control for performance optimization
8. **Automatic Registration**: Plugins work immediately after installation
9. **Robust**: Entry Points are more reliable than filesystem scanning
10. **Scalable**: System performance doesn't degrade with large numbers of installed plugins

## Technical Implementation Notes

### Entry Points vs Other Discovery Methods

**Why Entry Points?**
1. **Python Standard**: Official Python packaging mechanism for plugins
2. **Metadata-based**: No filesystem scanning required
3. **Reliable**: Works consistently across different Python environments
4. **Tool Support**: Supported by pip, setuptools, and other packaging tools
5. **Performance**: Fast discovery using package metadata

### Entry Points Discovery Process with Lazy Loading

1. **Package Installation**: When `pip install owa-env-example` runs:
   - Entry point metadata is stored in package metadata
   - No special file inclusion configuration needed

2. **Plugin Discovery**: On OWA startup:
   - System calls `pkg_resources.iter_entry_points('owa.env.plugins')`
   - Finds all packages that declared entry points in this group
   - Loads each plugin specification via the entry point

3. **Component Registration (Lazy)**: For each discovered plugin:
   - Parse plugin specification
   - Register component metadata only (import paths) with unified `namespace/name` naming
   - **No actual component imports occur** - components remain unloaded
   - Handle errors gracefully with warnings

4. **Component Retrieval (Lazy Loading)**: When components are accessed:
   - `CALLABLES["example/add"]` triggers import of the component
   - Component is cached for subsequent access
   - Import errors are handled at retrieval time

5. **Ready to Use**: Component metadata is immediately available, actual components loaded on demand

### Plugin Developer Workflow

1. **Create plugin code** with `plugin_spec` variable
2. **Declare entry point** in `pyproject.toml`
3. **Build and publish** package normally
4. **Users install** with `pip install`
5. **Components work immediately** - no activation needed

This approach is simpler, more reliable, and follows Python standards compared to filesystem-based discovery methods.