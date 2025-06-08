# Plugin Specification Guide

This guide covers how to write PluginSpec configurations in both Python and YAML formats for Open World Agents (OWA) plugins.

## Overview

A PluginSpec defines the structure and components of your plugin, enabling automatic discovery and registration through the OWA plugin system. You can define your plugin specification in two ways:

1. **Python Format** - Direct PluginSpec object (recommended for most cases)
2. **YAML Format** - External YAML file (useful for configuration-driven development)

## Python Format (Recommended)

### Basic Structure

Create a `PluginSpec` object in your plugin's `__init__.py` file:

```python
from owa.core.plugin_spec import PluginSpec

plugin_spec = PluginSpec(
    namespace="myplugin",
    version="1.0.0", 
    description="My custom plugin for OWA",
    author="Your Name",  # Optional
    components={
        "callables": {
            "function_name": "module.path:function_name",
        },
        "listeners": {
            "listener_name": "module.path:ListenerClass",
        },
        "runnables": {
            "runnable_name": "module.path:RunnableClass",
        }
    }
)
```

### Required Fields

- **namespace**: Unique identifier for your plugin (letters, numbers, underscores, hyphens)
- **version**: Plugin version following semantic versioning
- **description**: Brief description of plugin functionality
- **components**: Dictionary of component types and their definitions

### Optional Fields

- **author**: Plugin author name

### Component Types

#### Callables
Functions that users can call directly:

```python
"callables": {
    "add": "owa.env.myplugin.math:add_numbers",
    "screen.capture": "owa.env.myplugin.screen:capture_screen",
    "mouse.click": "owa.env.myplugin.input:click_mouse",
}
```

#### Listeners  
Event-driven components that respond to system events:

```python
"listeners": {
    "keyboard": "owa.env.myplugin.input:KeyboardListener",
    "file.watcher": "owa.env.myplugin.fs:FileWatcher",
    "network.monitor": "owa.env.myplugin.net:NetworkMonitor",
}
```

#### Runnables
Background processes that can be started and stopped:

```python
"runnables": {
    "data.processor": "owa.env.myplugin.processing:DataProcessor", 
    "log.collector": "owa.env.myplugin.logging:LogCollector",
    "health.monitor": "owa.env.myplugin.monitoring:HealthMonitor",
}
```

### Complete Example

```python
from owa.core.plugin_spec import PluginSpec

plugin_spec = PluginSpec(
    namespace="mycompany_tools",
    version="2.1.0",
    description="Custom tools for automation and monitoring",
    author="MyCompany Development Team",
    components={
        "callables": {
            # Math utilities
            "math.add": "owa.env.mycompany_tools.math:add_numbers",
            "math.multiply": "owa.env.mycompany_tools.math:multiply_numbers",
            
            # File operations
            "file.read": "owa.env.mycompany_tools.files:read_file",
            "file.write": "owa.env.mycompany_tools.files:write_file",
            
            # System utilities
            "system.info": "owa.env.mycompany_tools.system:get_system_info",
        },
        "listeners": {
            # Event monitoring
            "file.changes": "owa.env.mycompany_tools.monitoring:FileChangeListener",
            "system.alerts": "owa.env.mycompany_tools.monitoring:SystemAlertListener",
        },
        "runnables": {
            # Background services
            "log.processor": "owa.env.mycompany_tools.services:LogProcessor",
            "health.checker": "owa.env.mycompany_tools.services:HealthChecker",
        }
    }
)
```

### Entry Point Declaration

In your `pyproject.toml`, declare the entry point:

```toml
[project.entry-points."owa.env.plugins"]
mycompany_tools = "owa.env.mycompany_tools:plugin_spec"
```

## YAML Format

### Basic Structure

Create a `plugin.yaml` file with the same structure:

```yaml
namespace: myplugin
version: "1.0.0"
description: "My custom plugin for OWA"
author: "Your Name"  # Optional
components:
  callables:
    function_name: "module.path:function_name"
  listeners:
    listener_name: "module.path:ListenerClass"
  runnables:
    runnable_name: "module.path:RunnableClass"
```

### Complete YAML Example

```yaml
namespace: mycompany_tools
version: "2.1.0"
description: "Custom tools for automation and monitoring"
author: "MyCompany Development Team"
components:
  callables:
    # Math utilities
    math.add: "owa.env.mycompany_tools.math:add_numbers"
    math.multiply: "owa.env.mycompany_tools.math:multiply_numbers"
    
    # File operations  
    file.read: "owa.env.mycompany_tools.files:read_file"
    file.write: "owa.env.mycompany_tools.files:write_file"
    
    # System utilities
    system.info: "owa.env.mycompany_tools.system:get_system_info"
    
  listeners:
    # Event monitoring
    file.changes: "owa.env.mycompany_tools.monitoring:FileChangeListener"
    system.alerts: "owa.env.mycompany_tools.monitoring:SystemAlertListener"
    
  runnables:
    # Background services
    log.processor: "owa.env.mycompany_tools.services:LogProcessor"
    health.checker: "owa.env.mycompany_tools.services:HealthChecker"
```

### Loading YAML in Python

To use a YAML specification in your plugin:

```python
from owa.core.plugin_spec import PluginSpec
from pathlib import Path

# Load from YAML file
plugin_spec = PluginSpec.from_yaml(Path(__file__).parent / "plugin.yaml")
```

## Naming Conventions

### Namespace Rules
- Use letters, numbers, underscores, and hyphens only
- Keep it short and descriptive
- Examples: `desktop`, `gst`, `mycompany_tools`

### Component Name Rules  
- Use letters, numbers, underscores, and dots only
- Use dots for logical grouping: `mouse.click`, `file.read`
- Keep names descriptive and consistent
- Examples: `screen_capture`, `mouse.click`, `omnimodal.recorder`

### Import Path Format
- Must use format: `"module.path:object_name"`
- Module path should be importable Python module
- Object name should be the exact name in the module
- Examples: `"owa.env.myplugin.utils:helper_function"`

## Validation

### Using the CLI Tool

Validate your plugin specification:

```bash
# Validate Python entry point
owl env validate owa.env.myplugin:plugin_spec

# Validate YAML file  
owl env validate ./plugin.yaml

# Validate with verbose output
owl env validate owa.env.myplugin:plugin_spec --verbose

# Skip import validation (faster)
owl env validate ./plugin.yaml --no-check-imports
```

### Common Validation Errors

1. **Invalid namespace**: Contains invalid characters
2. **Invalid component names**: Contains invalid characters  
3. **Missing colon in import path**: Must use `module:object` format
4. **Module not found**: Import path points to non-existent module
5. **Object not found**: Object doesn't exist in specified module
6. **Wrong object type**: Object is not a PluginSpec instance (for entry points)

## Best Practices

### 1. Choose the Right Format
- **Python format**: Better for most cases, easier IDE support, type checking
- **YAML format**: Good for configuration-driven development, external tools

### 2. Organize Components Logically
```python
components={
    "callables": {
        # Group related functions
        "math.add": "...",
        "math.subtract": "...",
        "file.read": "...", 
        "file.write": "...",
    }
}
```

### 3. Use Descriptive Names
```python
# Good
"screen.capture": "owa.env.myplugin.screen:capture_screen"
"mouse.click": "owa.env.myplugin.input:click_mouse"

# Avoid
"sc": "owa.env.myplugin.screen:capture_screen"
"click": "owa.env.myplugin.input:click_mouse"
```

### 4. Keep Import Paths Consistent
```python
# Good - consistent module organization
"math.add": "owa.env.myplugin.math:add_numbers"
"math.subtract": "owa.env.myplugin.math:subtract_numbers"

# Avoid - scattered across modules
"math.add": "owa.env.myplugin.utils:add_numbers"  
"math.subtract": "owa.env.myplugin.helpers:subtract_numbers"
```

### 5. Version Your Plugin Properly
- Follow semantic versioning: `MAJOR.MINOR.PATCH`
- Update version when making changes
- Document breaking changes in major versions

## Quick Reference

### Validation Commands
```bash
# Validate Python entry point
owl env validate owa.env.myplugin:plugin_spec

# Validate YAML file
owl env validate ./plugin.yaml

# Validate with verbose output
owl env validate ./plugin.yaml --verbose

# Skip import validation (faster)
owl env validate ./plugin.yaml --no-check-imports
```

### Example Files
- **[Complete YAML Example](examples/example_plugin.yaml)** - Comprehensive YAML specification example

## Next Steps

- See [Custom Plugins Guide](custom_plugins.md) for complete plugin development workflow
- Check [YAML Plugin Guide](yaml_plugin_guide.md) for YAML-specific details
- Use `owl env validate` to test your specifications
- Explore [CLI Tools Guide](guide.md#cli-tools-for-plugin-management) for development tools
