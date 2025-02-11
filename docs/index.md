# Welcome to Open World Agents

Open World Agents is a powerful modular agent system that enables dynamic module registration and real-time event processing. This documentation will guide you through the system's architecture, features, and usage.

## Key Features

- **Dynamic Module Registration**: Modules can be registered and activated at runtime
- **Event-Driven Architecture**: Real-time event processing with listeners
- **Extensible Design**: Easy to add custom modules and extend functionality
- **Desktop Integration**: Built-in support for screen capture, window management, and input handling
- **Cross-Platform**: Works on Windows and macOS

## Quick Start

1. Install the required dependencies:

```bash
uv install --group dev
```

2. Set up your environment variables:

```bash
UV_PROJECT_ENVIRONMENT=(path to virtual environment)
GST_PLUGIN_PATH=(repository directory)/projects/owa-env-gst/gst-plugins
```

3. Import and use the core functionality:

```python
from owa.registry import CALLABLES, LISTENERS, activate_module

# Activate standard environment
activate_module("owa.env.std")

# Use registered functions
time_ns = CALLABLES["clock.time_ns"]()
print(f"Current time in nanoseconds: {time_ns}")
```

## Project Structure

```
open-world-agents/
├── projects/
│   ├── core/           # Core functionality
│   ├── data_collection/# Data collection agents
│   └── minecraft_env/  # Minecraft integration
├── docs/              # Documentation
└── README.md         # Project overview
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](contributing.md) for details on how to:

- Set up your development environment
- Submit bug reports
- Propose new features
- Create pull requests

## License

This project is released under the MIT License. See the [LICENSE](https://github.com/yourusername/open-world-agents/blob/main/LICENSE) file for details.
