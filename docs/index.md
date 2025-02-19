# Welcome to Open World Agents

Open World Agents is a powerful modular agent system that enables dynamic module registration and real-time event processing. This documentation will guide you through the system's architecture, features, and usage.

## Key Features

- **Dynamic Module Registration**: Modules can be registered and activated at runtime.
- **Event-Driven Architecture**: Real-time event processing with listeners.
- **Extensible Design**: Easy to add custom modules and extend functionality.
- **Desktop Integration**: Built-in support for screen capture, window management, and input handling.
- **Cross-Platform**: Works on Windows and macOS.

## Quick Start

1. **Install package managers**:
    - Follow the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).
    - Follow the [miniforge installation guide](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install) to install `conda` and `mamba`.

2. **Setup virtual environments**:
    - On Windows, use `cmd` to activate the environment.
    ```sh
    mamba env create -n owa -f projects/owa-env-gst/environment_detail.yml
    mamba activate owa
    ```
    - if you want to install conda packages in existing environment, run following:
    ```sh
    mamba env update --name (your-env-name-here) --file projects/owa-env-gst/environment_detail.yml
    ```

3. **Install required dependencies**:
    - Use `python vuv.py` instead of `uv` for all `uv` commands to prevent `uv` from separating virtual environments across sub-repositories in a mono-repo. Argument `--inexact` is needed to prevent `uv` from deleting non-dependency packages and `--extra envs` is needed to install OWA Env Plugins.
    ```sh
    python vuv.py sync --inexact --extra envs
    ```
    - To use raw `uv` binary, you must setup `UV_PROJECT_ENVIRONMENT` environment variable. see [here](https://docs.astral.sh/uv/configuration/environment/#uv_project_environment)

4. **Import and use core functionality**:
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
│   ├── owa-env-desktop/
│   ├── owa-env-example/
│   ├── owa-env-gst/
│   └── minecraft_env/  # Minecraft integration
├── docs/              # Documentation
└── README.md         # Project overview
```

## Contributing

We welcome contributions! Please see our Contributing Guide for details on how to:

- Set up your development environment.
- Submit bug reports.
- Propose new features.
- Create pull requests.

## License

This project is released under the MIT License. See the [LICENSE](https://github.com/yourusername/open-world-agents/blob/main/LICENSE) file for details.
