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

class EventListener:
    """Example event listener."""
    def configure(self, callback=None):
        self.callback = callback
        return self

    @property
    def session(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class DataProcessor:
    """Example data processor."""
    def configure(self, batch_size=100):
        self.batch_size = batch_size
        return self

    def start(self):
        print(f"Starting processor with batch size {self.batch_size}")
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

### Flexible Component Access

The system provides multiple ways to access components:

```python
from owa.core.registry import CALLABLES, get_component, list_components

# Method 1: Direct registry access (unified naming)
add_func = CALLABLES["example/add"]
result = add_func(5, 3)

# Method 2: Namespace + name access
add_func = get_component("callables", namespace="example", name="add")
result = add_func(5, 3)

# Method 3: Get all components in a namespace
example_callables = get_component("callables", namespace="example")
# Returns: {"add": <function>, "print": <function>}

# Method 4: List available components
all_callables = list_components("callables")
example_only = list_components("callables", namespace="example")
```

## Entry Points Discovery Implementation

### Plugin Discovery and Registration

The system automatically discovers plugins via Entry Points on startup:

```python
# Entry Points-based discovery and registration system
import pkg_resources
from typing import Dict
from owa.core.plugin_spec import PluginSpec
from owa.core.registry import CALLABLES, LISTENERS, RUNNABLES

class EntryPointPluginRegistry:
    """Discover and register plugins using Entry Points."""

    def __init__(self):
        self.discovered_plugins = {}
        self.auto_discover()

    def auto_discover(self):
        """Automatically discover and register all plugins via Entry Points."""
        plugins = self.discover_plugins()

        for plugin_name, spec in plugins.items():
            self.register_plugin_components(spec)
            self.discovered_plugins[plugin_name] = spec

    def discover_plugins(self) -> Dict[str, PluginSpec]:
        """Discover all plugins via Entry Points."""
        discovered = {}

        # Use pkg_resources to find all entry points in the 'owa.env.plugins' group
        for entry_point in pkg_resources.iter_entry_points('owa.env.plugins'):
            try:
                # Load the plugin specification
                plugin_spec = entry_point.load()

                if isinstance(plugin_spec, PluginSpec):
                    discovered[entry_point.name] = plugin_spec
                    print(f"Discovered plugin: {entry_point.name} v{plugin_spec.version}")
                else:
                    print(f"Warning: {entry_point.name} does not provide a valid PluginSpec")

            except Exception as e:
                print(f"Warning: Could not load plugin {entry_point.name}: {e}")

        return discovered

    def register_plugin_components(self, spec: PluginSpec):
        """Register all components from a plugin specification."""
        for component_type, components in spec.components.items():
            registry = self.get_registry(component_type)
            if not registry:
                continue

            for name, module_path in components.items():
                # Use unified namespace/name pattern
                full_name = f"{spec.namespace}/{name}"

                try:
                    # Import and register the component
                    component = self.import_component(module_path)
                    registry.register(full_name)(component)
                    print(f"Registered {component_type}: {full_name}")

                except Exception as e:
                    print(f"Warning: Could not register {full_name}: {e}")

    def get_registry(self, component_type: str):
        """Get the appropriate registry for component type."""
        registries = {
            "callables": CALLABLES,
            "listeners": LISTENERS,
            "runnables": RUNNABLES,
        }
        return registries.get(component_type)


    def import_component(self, module_path: str):
        """Import component from module path."""
        import importlib
        module_name, component_name = module_path.split(":")
        module = importlib.import_module(module_name)
        return getattr(module, component_name)


# Global entry point registry instance
entry_point_registry = EntryPointPluginRegistry()
```

### How Entry Points Discovery Works

1. **Plugin Declaration**: Plugins declare themselves in `pyproject.toml`:
   ```toml
   [project.entry-points."owa.env.plugins"]
   example = "owa.env.example:plugin_spec"
   ```

2. **Automatic Discovery**: On OWA startup, the system scans all installed packages for entry points in the `"owa.env.plugins"` group

3. **Plugin Loading**: For each entry point found:
   - Load the specified module (`owa.env.example`)
   - Get the specified object (`plugin_spec`)
   - Validate it's a `PluginSpec` instance

4. **Component Registration**: Register all components with unified `namespace/name` pattern

5. **Error Handling**: Log warnings for problematic plugins but continue discovery

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

### Phase 1: Entry Points-Based Discovery and Unified Naming (Next Steps)
- 🔄 Remove `activate_module()` requirement
- 🔄 Implement Entry Points-based plugin discovery on startup
- 🔄 Unify all component naming to `namespace/name` format
- 🔄 Add flexible component access API (`get_component`, `list_components`)
- 🔄 Update CLI tools to support new patterns

### Phase 2: Enhanced Developer Experience (Future)
- 📋 Plugin template generator (`owl env create`)
- 📋 Plugin validation tools
- 📋 Better documentation and examples
- 📋 Migration tools for existing plugins

### Key Design Changes

1. **Standard Plugin Discovery**: Use Python's standard Entry Points mechanism
2. **Zero Configuration**: No more `activate_module()` - plugins work immediately after `pip install`
3. **Unified Naming**: All components use `namespace/name` pattern consistently
4. **Flexible Access**: Multiple ways to access components (direct, by namespace, by type)
5. **Robust Discovery**: Entry Points provide reliable, metadata-based plugin discovery

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
4. **Better Discovery**: Easy to explore what's available
5. **Flexible Access**: Multiple ways to get components based on use case
6. **Automatic Registration**: Plugins work immediately after installation
7. **Robust**: Entry Points are more reliable than filesystem scanning

## Technical Implementation Notes

### Entry Points vs Other Discovery Methods

**Why Entry Points?**
1. **Python Standard**: Official Python packaging mechanism for plugins
2. **Metadata-based**: No filesystem scanning required
3. **Reliable**: Works consistently across different Python environments
4. **Tool Support**: Supported by pip, setuptools, and other packaging tools
5. **Performance**: Fast discovery using package metadata

### Entry Points Discovery Process

1. **Package Installation**: When `pip install owa-env-example` runs:
   - Entry point metadata is stored in package metadata
   - No special file inclusion configuration needed

2. **Plugin Discovery**: On OWA startup:
   - System calls `pkg_resources.iter_entry_points('owa.env.plugins')`
   - Finds all packages that declared entry points in this group
   - Loads each plugin specification via the entry point

3. **Component Registration**: For each discovered plugin:
   - Parse plugin specification
   - Register all components with unified `namespace/name` naming
   - Handle errors gracefully with warnings

4. **Ready to Use**: Components are immediately available in registries

### Plugin Developer Workflow

1. **Create plugin code** with `plugin_spec` variable
2. **Declare entry point** in `pyproject.toml`
3. **Build and publish** package normally
4. **Users install** with `pip install`
5. **Components work immediately** - no activation needed

This approach is simpler, more reliable, and follows Python standards compared to filesystem-based discovery methods.