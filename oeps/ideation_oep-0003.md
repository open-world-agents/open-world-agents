# Automatic Plugin Discovery and Unified Naming for OWA

## Vision: Zero-Configuration Plugin Ecosystem

Imagine a developer Bob who wants to create an environment plugin "example". Instead of manually calling `activate_module()`, the system automatically discovers and registers all installed plugins. Users can then access components using a unified naming convention and flexible access patterns.

## Unified Naming Convention

All components now use the same `namespace/name` pattern:

- **Callables**: `example/print`, `example/add`, `std/time_ns`
- **Listeners**: `example/events`, `std/tick`
- **Runnables**: `example/processor`, `std/monitor`

This eliminates the inconsistency between `namespace.name` (callables) and `namespace/name` (listeners/runnables).

## Plugin Specification Format

Bob defines his plugin using a simple YAML specification:

```yaml
# plugin.yaml
namespace: example
version: 0.1.0
description: Example environment plugin for Open World Agents
author: Bob's AI Solutions
homepage: https://github.com/bob/owa-env-example

# Unified component mapping - all use namespace/name
components:
  callables:
    print: "owa.env.example.callables:enhanced_print"
    add: "owa.env.example.callables:add_numbers"
    calculate: "owa.env.example.callables:calculator"

  listeners:
    events: "owa.env.example.listeners:EventListener"
    timer: "owa.env.example.listeners:TimerListener"

  runnables:
    processor: "owa.env.example.runnables:DataProcessor"
    counter: "owa.env.example.runnables:CounterRunnable"

  messages:
    event: "owa.env.example.messages:EventMessage"
```

### Alternative: Python-based Specification

```python
# plugin.py
from owa.core.plugin_spec import PluginSpec

plugin_spec = PluginSpec(
    namespace="example",
    version="0.1.0",
    description="Example environment plugin",
    author="Bob's AI Solutions",
    components={
        "callables": {
            "print": "owa.env.example.callables:enhanced_print",
            "add": "owa.env.example.callables:add_numbers",
        },
        "listeners": {
            "events": "owa.env.example.listeners:EventListener",
        },
        "runnables": {
            "processor": "owa.env.example.runnables:DataProcessor",
        },
        "messages": {
            "event": "owa.env.example.messages:EventMessage",
        }
    }
)
```

### Simple Package Structure

Bob's plugin follows the standard OWA plugin structure:

```
owa-env-example/
├── pyproject.toml                    # Package metadata (handled by pip)
├── README.md                         # Documentation
├── owa/
│   └── env/
│       └── example/
│           ├── __init__.py           # Plugin specification and activation
│           ├── plugin.yaml           # Plugin specification (included in package)
│           ├── callables.py          # Callable implementations
│           ├── listeners.py          # Listener implementations
│           ├── runnables.py          # Runnable implementations
│           └── messages.py           # Message type definitions
└── tests/                            # Tests (optional)
    ├── test_callables.py
    ├── test_listeners.py
    └── test_runnables.py
```

### Package Configuration for YAML Inclusion

To include `plugin.yaml` in the installed package, Bob needs to configure `pyproject.toml`:

```toml
# pyproject.toml
[project]
name = "owa-env-example"
version = "0.1.0"
description = "Example environment plugin for Open World Agents"
dependencies = ["owa-core"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["owa"]

# Include YAML files in the package
[tool.hatch.build.targets.wheel.force-include]
"owa/env/example/plugin.yaml" = "owa/env/example/plugin.yaml"
```

Or using setuptools approach:

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

[tool.setuptools.package-data]
"owa.env.example" = ["*.yaml", "*.yml"]
```

## Automatic Plugin Discovery

### Zero-Configuration Usage

Users install Bob's plugin using standard pip:

```bash
pip install owa-env-example
```

**No `activate_module()` needed!** The system automatically discovers and registers all installed plugins:

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
# Returns: {"add": <function>, "print": <function>, "calculate": <function>}

# Method 4: List available components
all_callables = list_components("callables")
example_only = list_components("callables", namespace="example")
```

## Automatic Discovery Implementation

### Plugin Discovery and Registration

The system automatically discovers and registers plugins on startup:

```python
# Enhanced discovery and registration system
import importlib
import pkgutil
from pathlib import Path
from typing import Dict, Optional
from owa.core.plugin_spec import PluginSpec
from owa.core.registry import CALLABLES, LISTENERS, RUNNABLES

class AutoPluginRegistry:
    """Automatically discover and register plugins."""

    def __init__(self):
        self.discovered_plugins = {}
        self.auto_discover()

    def auto_discover(self, namespace: str = "owa.env"):
        """Automatically discover and register all plugins."""
        plugins = self.discover_plugins(namespace)

        for plugin_name, spec in plugins.items():
            self.register_plugin_components(spec)
            self.discovered_plugins[plugin_name] = spec

    def discover_plugins(self, namespace: str = "owa.env") -> Dict[str, PluginSpec]:
        """Discover all installed plugins in the namespace."""
        discovered = {}

        try:
            ns_module = importlib.import_module(namespace)

            # Scan for both package and module plugins
            for finder, name, ispkg in pkgutil.iter_modules(ns_module.__path__):
                plugin_path = f"{namespace}.{name}"

                try:
                    # Try to load plugin specification
                    spec = self.load_plugin_spec(plugin_path, ispkg)
                    if spec:
                        discovered[spec.namespace] = spec

                except Exception as e:
                    # Log but don't fail discovery for individual plugins
                    print(f"Warning: Could not load plugin {plugin_path}: {e}")

        except ImportError:
            # Namespace doesn't exist, return empty
            pass

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
        module_name, component_name = module_path.split(":")
        module = importlib.import_module(module_name)
        return getattr(module, component_name)

    def load_plugin_spec(self, plugin_path: str, is_package: bool) -> Optional[PluginSpec]:
        """Load plugin specification from various sources."""
        if is_package:
            return self.load_package_plugin(plugin_path)
        else:
            return self.load_module_plugin(plugin_path)

    def load_package_plugin(self, plugin_path: str) -> Optional[PluginSpec]:
        """Load plugin from package structure."""
        try:
            module = importlib.import_module(plugin_path)

            # Method 1: Try programmatic specification first (most reliable)
            if hasattr(module, 'plugin_spec'):
                return module.plugin_spec

            # Method 2: Try YAML specification using importlib.resources
            try:
                import importlib.resources as resources

                # Try to read YAML from package resources
                for spec_file in ["plugin.yaml", "plugin.yml"]:
                    try:
                        if resources.is_resource(plugin_path, spec_file):
                            yaml_content = resources.read_text(plugin_path, spec_file)
                            return PluginSpec.from_yaml(yaml_content)
                    except (FileNotFoundError, AttributeError):
                        continue

            except ImportError:
                # Fallback for older Python versions
                if hasattr(module, '__file__') and module.__file__:
                    module_dir = Path(module.__file__).parent
                    for spec_file in ["plugin.yaml", "plugin.yml"]:
                        spec_path = module_dir / spec_file
                        if spec_path.exists():
                            return PluginSpec.from_file(spec_path)

        except Exception as e:
            print(f"Warning: Could not load plugin spec from {plugin_path}: {e}")
        return None

    def load_module_plugin(self, plugin_path: str) -> Optional[PluginSpec]:
        """Load plugin from module file."""
        try:
            module = importlib.import_module(plugin_path)
            if hasattr(module, 'plugin_spec'):
                return module.plugin_spec
        except Exception:
            pass
        return None

# Global auto-registry instance
auto_registry = AutoPluginRegistry()
```

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

### Phase 1: Automatic Discovery and Unified Naming (Next Steps)
- 🔄 Remove `activate_module()` requirement
- 🔄 Implement automatic plugin discovery on startup
- 🔄 Unify all component naming to `namespace/name` format
- 🔄 Add flexible component access API (`get_component`, `list_components`)
- 🔄 Update CLI tools to support new patterns

### Phase 2: Enhanced Developer Experience (Future)
- 📋 Plugin template generator (`owl env create`)
- 📋 Plugin validation tools
- 📋 Better documentation and examples
- 📋 Migration tools for existing plugins

### Key Design Changes

1. **Zero Configuration**: No more `activate_module()` - plugins work immediately after `pip install`
2. **Unified Naming**: All components use `namespace/name` pattern consistently
3. **Flexible Access**: Multiple ways to access components (direct, by namespace, by type)
4. **Automatic Discovery**: System automatically finds and registers all installed plugins

### Migration from Current System

**Before (Current OWA):**
```python
from owa.core.registry import CALLABLES, LISTENERS, activate_module

activate_module("owa.env.std")
activate_module("owa.env.example")

time_func = CALLABLES["clock.time_ns"]  # dot notation
listener = LISTENERS["example/events"]  # slash notation
```

**After (New System):**
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

1. **Simpler Usage**: No need to remember which plugins to activate
2. **Consistent Naming**: All components follow the same pattern
3. **Better Discovery**: Easy to explore what's available
4. **Flexible Access**: Multiple ways to get components based on use case
5. **Automatic Registration**: Plugins work immediately after installation

## Technical Implementation Notes

### YAML File Inclusion in Packages

For YAML specifications to work, plugin developers need to ensure the YAML files are included in their packages. There are several approaches:

**Option 1: Use Python specification (Recommended)**
```python
# owa/env/example/__init__.py
from owa.core.plugin_spec import PluginSpec

plugin_spec = PluginSpec(
    namespace="example",
    version="0.1.0",
    description="Example environment plugin",
    author="Bob's AI Solutions",
    components={
        "callables": {
            "print": "owa.env.example.callables:enhanced_print",
            "add": "owa.env.example.callables:add_numbers",
        },
        # ... other components
    }
)
```

**Option 2: Include YAML in package data**
```toml
# pyproject.toml
[tool.setuptools.package-data]
"owa.env.example" = ["*.yaml", "*.yml"]
```

**Option 3: Use importlib.resources for YAML access**
The discovery system uses `importlib.resources` to properly access YAML files from installed packages, which works correctly with pip-installed packages.

### Plugin Discovery Process

1. **Scan namespace**: System scans `owa.env` namespace for installed packages
2. **Load specifications**: For each package, try to load plugin specification:
   - First: Look for `plugin_spec` variable in `__init__.py`
   - Second: Look for `plugin.yaml`/`plugin.yml` using `importlib.resources`
3. **Register components**: Parse specification and register all components with unified naming
4. **Handle errors**: Log warnings for problematic plugins but continue discovery

This approach ensures that plugins work immediately after `pip install` without requiring manual activation.