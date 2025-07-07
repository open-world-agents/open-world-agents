# Custom EnvPlugin Development

Create your own environment plugins using the Entry Points-based discovery system. Plugins are automatically discovered when installed, requiring zero configuration from users.

## What are Entry Points?

Entry Points are a standard Python packaging mechanism that allows packages to advertise components that can be discovered and loaded by other packages. They're defined in your `pyproject.toml` file and enable automatic plugin discovery.

!!! info "Entry Points Documentation"
    For detailed information about Entry Points and plugin development, see:

    - [Entry Points Specification](https://packaging.python.org/en/latest/specifications/entry-points/)
    - [Creating and Discovering Plugins](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/)

When you install a plugin with entry points, OWA automatically discovers and registers all components without any manual configuration. This is what makes the `pip install` → immediate availability workflow possible.

!!! info "Module Structure Freedom"
    You have complete freedom in organizing your plugin code. The `owa.env.*` pattern used in examples is just a recommendation, not a requirement.

    **Valid entry point examples:**

    - `my_company.tools:plugin_spec` - Company structure
    - `custom_plugins:plugin_spec` - Simple structure
    - `owa.env.plugins.myplugin:plugin_spec` - Recommended OWA structure

!!! tip "Try the Example Plugin"
    If you want to test the example plugin that OWA provides:

    ```bash
    pip install -e projects/owa-env-example
    owl env list example
    ```

## Quick Start

=== "1. Copy Template"

    Start by copying the [owa-env-example](https://github.com/open-world-agents/open-world-agents/tree/main/projects/owa-env-example) directory as your template:

    ```bash
    cp -r owa-env-example owa-env-myplugin
    cd owa-env-myplugin
    ```

    The template contains:
    ```
    owa-env-example/
    ├── owa/env/example/          # Component implementations
    ├── owa/env/plugins/example.py # Plugin specification
    ├── pyproject.toml            # Entry point declaration
    ├── tests/                    # Test files
    └── README.md
    ```

=== "2. Entry Point"

    Update `pyproject.toml` to declare your plugin entry point:

    ```toml
    [project.entry-points."owa.env.plugins"]
    myplugin = "your.module.path:plugin_spec"
    ```

    **Examples of valid entry points:**
    ```toml
    # Recommended OWA structure
    myplugin = "owa.env.plugins.myplugin:plugin_spec"

    # Your own company structure
    myplugin = "my_company.tools.myplugin:plugin_spec"

    # Flat structure
    myplugin = "myplugin_package:plugin_spec"
    ```

=== "3. Plugin Spec"

    Create your plugin specification file. This defines what components your plugin provides:

    ```python
    """
    Plugin specification for MyPlugin.

    Keep this separate to avoid circular imports during discovery.
    """
    from owa.core.plugin_spec import PluginSpec

    plugin_spec = PluginSpec(
        namespace="myplugin",           # Unique namespace for your plugin
        version="0.1.0",
        description="My custom plugin",
        author="Your Name",             # Optional
        components={
            "callables": {
                "hello": "your.module:say_hello",
                "add": "your.module:add_numbers",
            },
            "listeners": {
                "events": "your.module:EventListener",
            },
            "runnables": {
                "processor": "your.module:DataProcessor",
            }
        }
    )
    ```

=== "4. Implement Components"

    Write your component implementations:

    ```python
    # Callable function
    def say_hello(name: str = "World") -> str:
        """Say hello to someone."""
        return f"Hello, {name}!"

    def add_numbers(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    # Listener class
    from owa.core import Listener

    class EventListener(Listener):
        def on_configure(self, callback, **kwargs):
            # Setup event handling logic
            self.callback = callback
            # Implementation details...

    # Runnable class
    from owa.core import Runnable

    class DataProcessor(Runnable):
        def on_configure(self, **kwargs):
            # Setup background processing
            # Implementation details...
    ```

=== "5. Install & Test"

    Install your plugin in development mode and test:

    ```bash
    # Install in development mode (if you're in the plugin directory)
    pip install -e .

    # Or from project root (adjust path to your plugin)
    pip install -e path/to/your-plugin

    # Verify plugin is discovered
    owl env list myplugin

    # Test components are available
    python -c "from owa.core import CALLABLES; print(CALLABLES['myplugin/hello']())"
    ```

## PluginSpec Reference

The `PluginSpec` class is the core of your plugin definition. It tells OWA what components your plugin provides and where to find them.

### Required Fields

!!! info "PluginSpec Fields"

    === "namespace"
        **Unique identifier for your plugin**

        - Must contain only letters, numbers, underscores, and hyphens
        - Should be short and descriptive
        - Examples: `desktop`, `gst`, `mycompany_tools`

        ```python
        namespace="myplugin"  # Components will be accessible as "myplugin/component_name"
        ```

    === "version"
        **Plugin version following semantic versioning**

        - Format: `MAJOR.MINOR.PATCH`
        - Update when making changes
        - Document breaking changes in major versions

        ```python
        version="1.2.3"
        ```

    === "description"
        **Brief description of plugin functionality**

        - Should clearly explain what the plugin does
        - Used in CLI tools and documentation

        ```python
        description="Desktop automation tools for screen capture and input"
        ```

    === "components"
        **Dictionary defining all plugin components**

        - Keys are component types: `callables`, `listeners`, `runnables`
        - Values are dictionaries mapping component names to import paths
        - Import paths use format: `"module.path:object_name"`

        ```python
        components={
            "callables": {
                "hello": "your.module:say_hello",
                "math.add": "your.module.math:add_numbers"
            },
            "listeners": {
                "events": "your.module:EventListener"
            },
            "runnables": {
                "processor": "your.module:DataProcessor"
            }
        }
        ```

### Optional Fields

- **author**: Plugin author name (string)

### Complete PluginSpec Example

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



### YAML Format Alternative

You can also define your plugin specification in YAML format:

```yaml
namespace: mycompany_tools
version: "2.1.0"
description: "Custom tools for automation and monitoring"
author: "MyCompany Development Team"
components:
  callables:
    math.add: "owa.env.mycompany_tools.math:add_numbers"
    file.read: "owa.env.mycompany_tools.files:read_file"
  listeners:
    file.changes: "owa.env.mycompany_tools.monitoring:FileChangeListener"
  runnables:
    log.processor: "owa.env.mycompany_tools.services:LogProcessor"
```

Then load it in Python:

```python
from owa.core.plugin_spec import PluginSpec
from pathlib import Path

# Load from YAML file
plugin_spec = PluginSpec.from_yaml(Path(__file__).parent / "plugin.yaml")
```

## Component Types

OWA plugins can provide three types of components:

!!! tip "Component Types Overview"

    === "Callables"
        **Functions that users invoke directly**

        - Synchronous operations that return immediate results
        - Can be functions or callable classes
        - Accessed via `CALLABLES["namespace/name"](args)`

        ```python
        def capture_screen() -> np.ndarray:
            """Capture the current screen."""
            # Implementation
            return screenshot_array

        def click_mouse(x: int, y: int, button: str = "left"):
            """Click mouse at coordinates."""
            # Implementation
        ```

    === "Listeners"
        **Event-driven components that respond to system events**

        - React to external events (keyboard, mouse, file changes, etc.)
        - Run user-provided callbacks when events occur
        - Must implement `on_configure()` method

        ```python
        from owa.core import Listener

        class KeyboardListener(Listener):
            def on_configure(self, callback, **kwargs):
                """Configure the listener with user callback."""
                self.callback = callback
                # Setup keyboard event monitoring

            def start(self):
                """Start listening for events."""
                # Begin event monitoring

            def stop(self):
                """Stop listening for events."""
                # Cleanup and stop monitoring
        ```

    === "Runnables"
        **Background tasks that can be started and stopped**

        - Long-running background processes
        - Can be started, stopped, and monitored
        - Must implement `on_configure()` method

        ```python
        from owa.core import Runnable

        class DataCollector(Runnable):
            def on_configure(self, output_file: str, interval: float = 1.0):
                """Configure the runnable with parameters."""
                self.output_file = output_file
                self.interval = interval

            def run(self):
                """Main background task logic."""
                while self.running:
                    # Collect and save data
                    time.sleep(self.interval)
        ```

## Package Structure Examples

You have complete freedom in organizing your plugin code. Here are common patterns:

!!! example "Structure Options"

    === "Recommended OWA Structure"
        ```
        owa/env/plugins/myplugin.py     # Plugin specification
        owa/env/myplugin/               # Component implementations
        ├── __init__.py
        ├── callables.py                # Callable functions
        ├── listeners.py                # Listener classes
        └── runnables.py                # Runnable classes
        ```

        **Entry point:** `owa.env.plugins.myplugin:plugin_spec`

        This follows the same pattern as official OWA plugins.

    === "Company Structure"
        ```
        my_company/
        ├── tools/
        │   ├── plugin_spec.py          # Plugin specification
        │   ├── ai_processor.py         # AI-related components
        │   ├── data_analyzer.py        # Data analysis components
        │   └── utils.py                # Shared utilities
        └── config/
            └── settings.py
        ```

        **Entry point:** `my_company.tools.plugin_spec:plugin_spec`

        Good for company-internal plugins with existing code organization.

    === "Flat Structure"
        ```
        myplugin_package/
        ├── __init__.py                 # Plugin spec can be here
        ├── features.py                 # Main functionality
        ├── utils.py                    # Helper functions
        └── tests/
            └── test_features.py
        ```

        **Entry point:** `myplugin_package:plugin_spec`

        Simple structure for smaller plugins.



## Plugin Validation

Use the CLI to validate your plugin during development:

```bash
# Validate plugin specification (adjust path to match your structure)
owl env validate owa.env.plugins.myplugin:plugin_spec        # OWA structure
owl env validate my_company.tools.plugin_spec:plugin_spec   # Company structure
owl env validate myplugin_package:plugin_spec               # Flat structure

# Example with the example plugin
$ owl env validate owa.env.plugins.example:plugin_spec
✅ Plugin Specification Valid
├── Source: Entry point: owa.env.plugins.example:plugin_spec
├── 📋 Plugin Metadata
│   ├── ├── Namespace: example
│   ├── ├── Version: 0.1.0
│   ├── ├── Author: OWA Development Team
│   └── └── Description: Example environment plugin demonstrating the plugin system
└── 🔧 Components Summary
    ├── ├── Total Components: 6
    ├── ├── 📞 Callables: 2
    ├── ├── 👂 Listeners: 2
    └── └── 🏃 Runnables: 2

✅ Validation successful!

# Validate with detailed output
owl env validate your.module.path:plugin_spec --verbose

# Validate YAML specification (if using YAML format)
owl env validate ./plugin.yaml
```

## CLI Tools for Development

The `owl env` command provides comprehensive tools for plugin development:

!!! tip "Essential Commands"

    ```bash
    # Discovery and listing
    owl env list myplugin                          # List your plugin components (auto-shows details)
    owl env namespaces                             # See all available namespaces

    # Example with the example plugin
    $ owl env list example
    📦 Plugin: example (6 components)
    ├── 📞 Callables: 2
    ├── 👂 Listeners: 2
    └── 🏃 Runnables: 2
    📞 Callables (2)
    ├── example/add
    └── example/print
    👂 Listeners (2)
    ├── example/listener
    └── example/timer
    🏃 Runnables (2)
    ├── example/counter
    └── example/runnable

    # Component inspection
    owl env list myplugin --inspect my_function    # Inspect specific component
    owl env list myplugin --search my_function     # Search within your plugin

    # Ecosystem analysis
    owl env stats --by-namespace                   # Statistics by namespace
    owl env stats --namespaces                     # Show available namespaces
    ```

!!! info "Complete CLI Reference"
    For detailed information about all CLI commands and options:

    - **[CLI Tools](../cli/index.md)** - Complete command overview
    - **[Environment Commands](../cli/env.md)** - Detailed `owl env` documentation

## Publishing Your Plugin

Once your plugin is ready, you can use any Python packaging tool to build and publish it. OWA doesn't impose any restrictions on build backends or publishing methods.

**Recommended approach using uv:**

```bash
# Build your plugin
uv build

# Publish to PyPI
uv publish
```

**Alternative approaches:**

```bash
# Using build + twine
python -m build
python -m twine upload dist/*

# Using poetry
poetry build
poetry publish

# Using setuptools directly
python setup.py sdist bdist_wheel
twine upload dist/*
```

For detailed information on the recommended approach, see:

- [uv build documentation](https://docs.astral.sh/uv/guides/publish/#building-your-package)
- [uv publish documentation](https://docs.astral.sh/uv/guides/publish/#publishing-your-package)

Users can then install your plugin and components are automatically available:

```bash
pip install your-plugin-name
```

```python
from owa.core import CALLABLES
result = CALLABLES["myplugin/hello"]("World")
```




