# How to write your own EnvPlugin

You can write & contribute your own EnvPlugin using the Entry Points-based system for automatic discovery.

> **🚨 CRITICAL: Module Structure is COMPLETELY FLEXIBLE**
>
> **The `owa.env.*` structure shown in examples is just a RECOMMENDATION, NOT a requirement!**
>
> - ✅ You can use ANY module structure: `my_company.tools`, `custom_plugins`, `anything.you.want`
> - ✅ You can organize your code however makes sense for your project
> - ✅ The ONLY requirement is that your entry point correctly points to your plugin specification
> - ❌ You are NOT required to follow the `owa.env.*` pattern
>
> **Examples of valid entry points:**
> - `my_company.ai_tools.plugin_spec:plugin_spec`
> - `custom_plugins:plugin_spec`
> - `automation.workflows.owa_integration:plugin_spec`
> - `owa.env.plugins.myplugin:plugin_spec` (recommended but not required)

## Quick Start

1. **Copy the example template**: Copy & Paste [owa-env-example](https://github.com/open-world-agents/open-world-agents/tree/main/projects/owa-env-example) directory. This directory contains:
    ```sh
    owa-env-example
    ├── owa/env/example
    │   ├── __init__.py           # Main module (no plugin_spec)
    │   ├── example_callable.py
    │   ├── example_listener.py
    │   └── example_runnable.py
    ├── owa/env/plugins
    │   └── example.py            # Plugin specification
    ├── pyproject.toml            # Entry point declaration
    ├── README.md
    ├── tests
    │   └── test_print.py
    └── uv.lock
    ```

2. **Rename and customize**: Rename `owa-env-example` to your plugin name (e.g., `owa-env-myplugin`).

3. **Update Entry Point Declaration**: In `pyproject.toml`, update the entry point (you can use ANY module path you prefer):
    ```toml
    [project.entry-points."owa.env.plugins"]
    # Examples - choose ANY structure you like:
    myplugin = "owa.env.plugins.myplugin:plugin_spec"     # Recommended OWA structure
    # myplugin = "my_company.tools.myplugin:plugin_spec"  # Your own structure
    # myplugin = "myplugin_spec:plugin_spec"              # Flat structure
    # myplugin = "plugins.custom.myplugin:plugin_spec"    # Custom hierarchy
    ```

4. **Create Plugin Specification**: Create your plugin specification file (location is flexible - this is just one example):
    ```python
    """
    Plugin specification for the MyPlugin environment plugin.

    This module is kept separate to avoid circular imports during plugin discovery.
    """

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

6. **Package Structure**: **COMPLETE FREEDOM** - organize your plugin however you want!
    - **Plugin Specification**: Can be located ANYWHERE in your module structure
    - **Module Structure**: **ZERO limitations** - use any organization that makes sense for your project
    - **Entry Point Registration**: The ONLY requirement is that your entry point correctly points to your plugin specification

    **The `owa.env.*` structure is just a RECOMMENDATION used by official OWA plugins. You are NOT required to follow it!**

    **Example structures (ALL completely valid - choose what works for you!)**:
    ```
    # Option 1: Recommended OWA structure (but not required!)
    owa/env/plugins/myplugin.py     # Plugin specification
    owa/env/myplugin/               # Your implementation
    ├── __init__.py
    └── components.py

    # Option 2: Your own company structure
    my_company/tools/
    ├── plugin_spec.py              # Plugin specification
    ├── ai_processor.py
    └── data_analyzer.py
    # Entry point: my_company.tools.plugin_spec:plugin_spec

    # Option 3: Flat structure
    myplugin_package/
    ├── __init__.py                 # Plugin specification here
    ├── features.py
    └── utils.py
    # Entry point: myplugin_package:plugin_spec

    # Option 4: Domain-driven structure
    automation/workflows/
    ├── owa_integration.py          # Plugin specification
    ├── core/
    │   ├── engine.py
    │   └── config.py
    └── integrations/
        ├── api.py
        └── database.py
    # Entry point: automation.workflows.owa_integration:plugin_spec

    # Option 5: Organized by component type (within your own structure)
    custom_plugins/myplugin/
    ├── plugin_def.py               # Plugin specification
    ├── callables/
    │   ├── math.py
    │   └── utils.py
    ├── listeners/
    │   └── events.py
    └── runnables/
        └── processors.py
    # Entry point: custom_plugins.myplugin.plugin_def:plugin_spec
    ```

    **The ONLY requirement**:
    - Your entry point in `pyproject.toml` must correctly point to your plugin specification
    - That's it! No other structural requirements exist.

7. **Install and Test**: Install your plugin with `pip install -e .` and test that components are automatically available.

8. **Validate Plugin**: Use the CLI to validate your plugin specification (adjust the path to match YOUR structure):
   ```bash
   # Examples - use the path that matches YOUR entry point:
   owl env validate owa.env.plugins.myplugin:plugin_spec        # OWA recommended structure
   owl env validate my_company.tools.plugin_spec:plugin_spec   # Your own structure
   owl env validate myplugin_package:plugin_spec               # Flat structure
   owl env validate custom_plugins.myplugin.plugin_def:plugin_spec  # Custom structure

   # Validate YAML specification (if using YAML format)
   owl env validate ./plugin.yaml

   # Validate with detailed output (use YOUR entry point path)
   owl env validate your.module.path:plugin_spec --verbose

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

# Validate plugin specifications (use YOUR entry point path)
$ owl env validate owa.env.plugins.myplugin:plugin_spec    # OWA structure example
$ owl env validate my_company.tools.plugin_spec:plugin_spec # Your structure example
$ owl env validate ./plugin.yaml                           # YAML file
```

## Key Benefits of Entry Points System

- **Zero Configuration**: Users just `pip install` your plugin - no manual activation needed
- **Automatic Discovery**: Components are immediately available after installation
- **Unified Naming**: All components use `namespace/name` pattern for consistency
- **Python Standards**: Follows official Python packaging guidelines
- **Lazy Loading**: Components are imported only when accessed for better performance
- **CLI Support**: Rich command-line tools for plugin management and validation




