# How to write your own EnvPlugin

You can write & contribute your own EnvPlugin using the Entry Points-based system for automatic discovery.

## Quick Start

1. **Copy the example template**: Copy & Paste [owa-env-example](https://github.com/open-world-agents/open-world-agents/tree/main/projects/owa-env-example) directory. This directory contains:
    ```sh
    owa-env-example
    ├── owa/env/example
    │   ├── __init__.py           # Plugin specification
    │   ├── example_callable.py
    │   ├── example_listener.py
    │   └── example_runnable.py
    ├── pyproject.toml            # Entry point declaration
    ├── README.md
    ├── tests
    │   └── test_print.py
    └── uv.lock
    ```

2. **Rename and customize**: Rename `owa-env-example` to your plugin name (e.g., `owa-env-myplugin`).

3. **Update Entry Point Declaration**: In `pyproject.toml`, update the entry point:
    ```toml
    [project.entry-points."owa.env.plugins"]
    myplugin = "owa.env.myplugin:plugin_spec"
    ```

4. **Create Plugin Specification**: In your `__init__.py`, define the plugin specification:
    ```python
    from owa.core.plugin_spec import PluginSpec

    plugin_spec = PluginSpec(
        namespace="myplugin",
        version="0.1.0",
        description="My custom plugin",
        author="Your Name",
        components={
            "callables": {
                "hello": "owa.env.myplugin:say_hello",
                "add": "owa.env.myplugin:add_numbers",
            },
            "listeners": {
                "events": "owa.env.myplugin:EventListener",
            },
            "runnables": {
                "processor": "owa.env.myplugin:DataProcessor",
            }
        }
    )
    ```

    **📖 For detailed guidance on writing plugin specifications, see:**
    - **[Plugin Specification Guide](plugin_specification_guide.md)** - Complete guide for Python and YAML formats
    - **[YAML Plugin Guide](yaml_plugin_guide.md)** - Focused guide for YAML-based specifications

5. **Implement Components**: Write your component implementations using the unified `namespace/name` pattern.

6. **Package Structure**: Maintain the [namespace package](https://packaging.python.org/en/latest/guides/packaging-namespace-packages/) structure:
    - **Important**: All source files must be inside the `owa/env/myplugin` folder.
    - **What NOT to do**: Don't place source files between `owa` and `owa/env/myplugin`.
    - **Correct structure**:
        ```
        owa
        └── env
            └── myplugin
                ├── __init__.py      # Plugin specification
                ├── your_code.py
                ├── your_module.py
                └── components.py
        ```

7. **Install and Test**: Install your plugin with `pip install -e .` and test that components are automatically available.

8. **Validate Plugin**: Use the CLI to validate your plugin specification:
   ```bash
   # Validate Python entry point specification
   owl env validate owa.env.myplugin:plugin_spec

   # Validate YAML specification (if using YAML format)
   owl env validate ./plugin.yaml

   # Validate with detailed output
   owl env validate owa.env.myplugin:plugin_spec --verbose

   # List your plugin to verify it's discovered
   owl env list --namespace myplugin

   # Show detailed component information
   owl env show myplugin --components
   ```

9. **Contribute**: Make a PR following the [Contributing Guide](../contributing.md).

## CLI Tools for Plugin Development

The `owl env` command provides comprehensive tools for plugin development and testing:

```bash
# Discover and list your plugin
$ owl env list --namespace myplugin

# Show detailed component information with import paths
$ owl env show myplugin --components --details

# Inspect specific components
$ owl env show myplugin --inspect my_function

# Search for components in your plugin
$ owl env search "my.*function" --namespace myplugin

# List specific component types with details
$ owl env list --type callables --details --table

# Check ecosystem health and your plugin's integration
$ owl env health
$ owl env stats --by-namespace

# Quick exploration shortcuts
$ owl env ls myplugin                              # Quick plugin overview
$ owl env find my_function                         # Quick component search
$ owl env namespaces                               # See all available namespaces

# Validate plugin specifications
$ owl env validate owa.env.myplugin:plugin_spec    # Python entry point
$ owl env validate ./plugin.yaml                   # YAML file
```

## Key Benefits of Entry Points System

- **Zero Configuration**: Users just `pip install` your plugin - no manual activation needed
- **Automatic Discovery**: Components are immediately available after installation
- **Unified Naming**: All components use `namespace/name` pattern for consistency
- **Python Standards**: Follows official Python packaging guidelines
- **Lazy Loading**: Components are imported only when accessed for better performance
- **CLI Support**: Rich command-line tools for plugin management and validation




