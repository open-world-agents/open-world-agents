# YAML Plugin Specification Guide

This guide focuses specifically on creating and managing plugin specifications using YAML format for Open World Agents (OWA).

## Why Use YAML Format?

YAML format is ideal when you want to:
- Separate configuration from code
- Enable non-programmers to modify plugin definitions
- Use external tools to generate or modify plugin specifications
- Version control plugin configurations separately from implementation
- Support dynamic plugin configuration

## Basic YAML Structure

### Minimal Example

```yaml
namespace: simple_plugin
version: "1.0.0"
description: "A simple example plugin"
components:
  callables:
    hello: "owa.env.simple_plugin:say_hello"
```

### Complete Example

```yaml
namespace: advanced_plugin
version: "2.1.0"
description: "Advanced plugin with multiple component types"
author: "Plugin Developer"
components:
  callables:
    # Math operations
    add: "owa.env.advanced_plugin.math:add_numbers"
    multiply: "owa.env.advanced_plugin.math:multiply_numbers"
    
    # File operations
    file.read: "owa.env.advanced_plugin.files:read_file"
    file.write: "owa.env.advanced_plugin.files:write_file"
    file.exists: "owa.env.advanced_plugin.files:file_exists"
    
    # System utilities
    system.info: "owa.env.advanced_plugin.system:get_system_info"
    process.list: "owa.env.advanced_plugin.system:list_processes"
    
  listeners:
    # Event monitoring
    file.watcher: "owa.env.advanced_plugin.monitoring:FileWatcher"
    keyboard.events: "owa.env.advanced_plugin.input:KeyboardListener"
    network.monitor: "owa.env.advanced_plugin.network:NetworkMonitor"
    
  runnables:
    # Background services
    log.processor: "owa.env.advanced_plugin.services:LogProcessor"
    data.collector: "owa.env.advanced_plugin.services:DataCollector"
    health.monitor: "owa.env.advanced_plugin.services:HealthMonitor"
```

## Field Reference

### Required Fields

#### namespace
- **Type**: String
- **Rules**: Letters, numbers, underscores, hyphens only
- **Purpose**: Unique identifier for your plugin
- **Examples**: `desktop`, `gst`, `my_company_tools`

```yaml
namespace: my_awesome_plugin
```

#### version
- **Type**: String (quoted recommended)
- **Rules**: Follow semantic versioning
- **Purpose**: Plugin version tracking
- **Examples**: `"1.0.0"`, `"2.1.3"`, `"0.1.0-beta"`

```yaml
version: "1.2.3"
```

#### description
- **Type**: String
- **Purpose**: Brief description of plugin functionality
- **Best Practice**: Keep under 100 characters

```yaml
description: "Desktop automation tools for mouse, keyboard, and window management"
```

#### components
- **Type**: Dictionary
- **Purpose**: Defines all plugin components
- **Required**: At least one component type

```yaml
components:
  callables: {}    # At least one of these
  listeners: {}    # must be present
  runnables: {}    # (can be empty)
```

### Optional Fields

#### author
- **Type**: String
- **Purpose**: Plugin author information

```yaml
author: "John Doe <john@example.com>"
```

## Component Types

### Callables
Functions that users call directly to perform actions or get information.

```yaml
components:
  callables:
    # Simple function
    hello: "owa.env.myplugin:say_hello"
    
    # Grouped functions using dots
    math.add: "owa.env.myplugin.math:add_numbers"
    math.subtract: "owa.env.myplugin.math:subtract_numbers"
    
    # Complex nested grouping
    file.text.read: "owa.env.myplugin.files:read_text_file"
    file.binary.read: "owa.env.myplugin.files:read_binary_file"
    
    # System operations
    system.memory.usage: "owa.env.myplugin.system:get_memory_usage"
    system.cpu.usage: "owa.env.myplugin.system:get_cpu_usage"
```

### Listeners
Event-driven components that respond to system events with user-provided callbacks.

```yaml
components:
  listeners:
    # Input monitoring
    keyboard: "owa.env.myplugin.input:KeyboardListener"
    mouse: "owa.env.myplugin.input:MouseListener"
    
    # File system monitoring
    file.changes: "owa.env.myplugin.fs:FileChangeListener"
    directory.watcher: "owa.env.myplugin.fs:DirectoryWatcher"
    
    # Network monitoring
    network.traffic: "owa.env.myplugin.network:TrafficListener"
    connection.monitor: "owa.env.myplugin.network:ConnectionMonitor"
    
    # System events
    process.monitor: "owa.env.myplugin.system:ProcessMonitor"
    window.events: "owa.env.myplugin.ui:WindowEventListener"
```

### Runnables
Background processes that can be started, stopped, and managed.

```yaml
components:
  runnables:
    # Data processing
    log.processor: "owa.env.myplugin.processing:LogProcessor"
    data.analyzer: "owa.env.myplugin.processing:DataAnalyzer"
    
    # Monitoring services
    health.checker: "owa.env.myplugin.monitoring:HealthChecker"
    performance.monitor: "owa.env.myplugin.monitoring:PerformanceMonitor"
    
    # Background tasks
    backup.service: "owa.env.myplugin.services:BackupService"
    cleanup.task: "owa.env.myplugin.services:CleanupTask"
```

## Import Path Format

All component values must follow the format: `"module.path:object_name"`

### Valid Examples
```yaml
components:
  callables:
    # Standard format
    function1: "owa.env.myplugin.module:function_name"
    
    # Deep module path
    function2: "owa.env.myplugin.subpackage.module:function_name"
    
    # Class reference
    processor: "owa.env.myplugin.processing:DataProcessor"
    
    # Function in __init__.py
    helper: "owa.env.myplugin:helper_function"
```

### Invalid Examples
```yaml
components:
  callables:
    # Missing colon
    bad1: "owa.env.myplugin.module.function_name"
    
    # Wrong separator
    bad2: "owa.env.myplugin.module::function_name"
    
    # No module path
    bad3: "function_name"
    
    # Multiple colons
    bad4: "owa.env.myplugin:module:function"
```

## Loading YAML in Python

### Method 1: Direct Loading
```python
from owa.core.plugin_spec import PluginSpec
from pathlib import Path

# Load from YAML file
plugin_spec = PluginSpec.from_yaml("plugin.yaml")

# Or with Path object
plugin_spec = PluginSpec.from_yaml(Path(__file__).parent / "plugin.yaml")
```

### Method 2: Entry Point with YAML
```python
# In your __init__.py
from owa.core.plugin_spec import PluginSpec
from pathlib import Path

# Load the YAML specification
plugin_spec = PluginSpec.from_yaml(Path(__file__).parent / "plugin.yaml")
```

Then in `pyproject.toml`:
```toml
[project.entry-points."owa.env.plugins"]
myplugin = "owa.env.myplugin:plugin_spec"
```

## Real-World Examples

### Example 1: Development Tools Plugin
```yaml
namespace: devtools
version: "1.0.0"
description: "Development utilities and automation tools"
author: "DevTeam"
components:
  callables:
    # Git operations
    git.status: "owa.env.devtools.git:get_status"
    git.commit: "owa.env.devtools.git:create_commit"
    git.push: "owa.env.devtools.git:push_changes"
    
    # Code analysis
    code.lint: "owa.env.devtools.analysis:lint_code"
    code.format: "owa.env.devtools.analysis:format_code"
    code.test: "owa.env.devtools.analysis:run_tests"
    
    # Build operations
    build.compile: "owa.env.devtools.build:compile_project"
    build.package: "owa.env.devtools.build:package_project"
    
  listeners:
    # File monitoring
    file.changes: "owa.env.devtools.monitoring:FileChangeListener"
    test.runner: "owa.env.devtools.testing:TestRunnerListener"
    
  runnables:
    # Background services
    auto.formatter: "owa.env.devtools.services:AutoFormatter"
    continuous.tester: "owa.env.devtools.services:ContinuousTester"
```

### Example 2: System Monitoring Plugin
```yaml
namespace: sysmon
version: "2.0.1"
description: "Comprehensive system monitoring and alerting"
author: "SysAdmin Team <sysadmin@company.com>"
components:
  callables:
    # Resource monitoring
    cpu.usage: "owa.env.sysmon.resources:get_cpu_usage"
    memory.usage: "owa.env.sysmon.resources:get_memory_usage"
    disk.usage: "owa.env.sysmon.resources:get_disk_usage"
    network.stats: "owa.env.sysmon.resources:get_network_stats"
    
    # Process management
    process.list: "owa.env.sysmon.processes:list_processes"
    process.kill: "owa.env.sysmon.processes:kill_process"
    process.info: "owa.env.sysmon.processes:get_process_info"
    
    # System information
    system.info: "owa.env.sysmon.system:get_system_info"
    uptime: "owa.env.sysmon.system:get_uptime"
    
  listeners:
    # Resource monitoring
    cpu.monitor: "owa.env.sysmon.monitoring:CpuMonitor"
    memory.monitor: "owa.env.sysmon.monitoring:MemoryMonitor"
    disk.monitor: "owa.env.sysmon.monitoring:DiskMonitor"
    
    # Process monitoring
    process.monitor: "owa.env.sysmon.monitoring:ProcessMonitor"
    service.monitor: "owa.env.sysmon.monitoring:ServiceMonitor"
    
  runnables:
    # Alert services
    alert.manager: "owa.env.sysmon.alerts:AlertManager"
    log.collector: "owa.env.sysmon.logging:LogCollector"
    metric.collector: "owa.env.sysmon.metrics:MetricCollector"
```

## Validation and Testing

### Using the CLI Tool
```bash
# Validate YAML file
owl env validate plugin.yaml

# Validate with detailed output
owl env validate plugin.yaml --verbose

# Skip import validation for faster checking
owl env validate plugin.yaml --no-check-imports

# Validate and show component details
owl env validate plugin.yaml --check-imports
```

### Common YAML Errors

1. **Syntax Errors**
```yaml
# Wrong: Missing quotes around version
version: 1.0.0

# Correct: Version should be quoted
version: "1.0.0"
```

2. **Indentation Errors**
```yaml
# Wrong: Inconsistent indentation
components:
  callables:
  hello: "module:function"

# Correct: Consistent indentation
components:
  callables:
    hello: "module:function"
```

3. **Invalid Import Paths**
```yaml
# Wrong: Missing colon
components:
  callables:
    hello: "owa.env.myplugin.say_hello"

# Correct: Include colon separator
components:
  callables:
    hello: "owa.env.myplugin:say_hello"
```

## Best Practices

### 1. File Organization
```
owa-env-myplugin/
├── owa/env/myplugin/
│   ├── __init__.py          # Load plugin_spec from YAML
│   ├── plugin.yaml          # Plugin specification
│   ├── core.py             # Core functionality
│   └── utils.py            # Utility functions
├── pyproject.toml          # Entry point declaration
└── README.md
```

### 2. YAML Structure
```yaml
# Use comments to organize sections
namespace: myplugin
version: "1.0.0"
description: "Plugin description"
author: "Author Name"

components:
  # Core functionality
  callables:
    # Group related functions
    math.add: "owa.env.myplugin.math:add"
    math.subtract: "owa.env.myplugin.math:subtract"
    
    # File operations
    file.read: "owa.env.myplugin.files:read"
    file.write: "owa.env.myplugin.files:write"
  
  # Event handling
  listeners:
    events: "owa.env.myplugin.events:EventListener"
  
  # Background services  
  runnables:
    processor: "owa.env.myplugin.services:Processor"
```

### 3. Version Control
- Keep YAML files in version control
- Use meaningful commit messages for specification changes
- Tag releases with version numbers matching the YAML version

### 4. Documentation
- Document each component in comments
- Maintain a changelog for specification changes
- Include examples in your README

## Converting Between Formats

### Python to YAML
```python
from owa.core.plugin_spec import PluginSpec

# Existing Python specification
plugin_spec = PluginSpec(
    namespace="myplugin",
    version="1.0.0",
    description="My plugin",
    components={"callables": {"hello": "owa.env.myplugin:say_hello"}}
)

# Save to YAML
plugin_spec.to_yaml("plugin.yaml")
```

### YAML to Python
```python
from owa.core.plugin_spec import PluginSpec

# Load from YAML
plugin_spec = PluginSpec.from_yaml("plugin.yaml")

# Now you can use it as a Python object
print(plugin_spec.namespace)
print(plugin_spec.components)
```

## Example Files

- **[Complete YAML Example](examples/example_plugin.yaml)** - Comprehensive example demonstrating all features

## Next Steps

- See [Plugin Specification Guide](plugin_specification_guide.md) for Python format details
- Check [Custom Plugins Guide](custom_plugins.md) for complete development workflow
- Use `owl env validate` to test your YAML specifications
- Explore [CLI Tools](guide.md#cli-tools-for-plugin-management) for development assistance
