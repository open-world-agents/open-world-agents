# Welcome to Open World Agents

### Everything you need to build state-of-the-art foundation multimodal desktop agent, end-to-end.

Streamline your agent's lifecycle with Open World Agents. From data capture to real-time evaluation, everything is designed for flexibility and performance.

With open-world-agents, you can:

- **Data Collection**: Collect your own desktop data, which contains timestamp-aligned **keyboard/mouse control** and **high-frequency(60Hz+) screen data**.
    - Powered by Windows APIs (`DXGI/WGC`) and the robust [GStreamer](https://gstreamer.freedesktop.org/) framework, ensuring superior performance compared to alternatives. [Learn more...](recorder/why.md)
- **Asynchronous, real-time event processing**: Compared to existing LLM-agent frameworks and [gymnasium.Env](https://gymnasium.farama.org/api/env/), our platform features an asynchronous processing design leveraging `Callables`, `Listeners`, and `Runnables`. [Learn more...](env/architecture.md)
- **Dynamic EnvPlugin Registration**: Seamlessly register and activate EnvPlugins—consisting of `Callables`, `Listeners`, and `Runnables`—at runtime to customize and extend functionality. [Learn more...](env/install_and_usage.md)
- **Extensible Design**: Easy to add custom EnvPlugin and extend functionality. [Learn more...](env/custom_plugins.md)
- **Comprehensive Examples**: We provides various examples that demonstrates how to build foundation multimodal desktop agent. Since it's just a example, you may customize anything you want.
<!-- - **Cross-Platform**: Works on Windows and macOS. -->

## Quick Start

1. **Install package managers, uv and conda**:
    - Follow the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).
    - Follow the [miniforge installation guide](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install) to install `conda` and `mamba`.

2. **Setup virtual environments**:
    - (Recommended) Create new environment with dependencies.
    ```sh
    mamba env create -n owa -f projects/owa-env-gst/environment_detail.yml
    conda activate owa
    ```
    - If you want to install conda packages in existing environment, run following:
    ```sh
    mamba env update --name (your-env-name-here) --file projects/owa-env-gst/environment_detail.yml
    ```

3. **Install required dependencies**:
    - Use `python vuv.py` instead of `uv` for all `uv` commands to prevent `uv` from separating virtual environments across sub-repositories in a mono-repo. Argument `--inexact` is needed to prevent `uv` from deleting non-dependency packages and `--extra envs` is needed to install EnvPlugin.
    ```sh
    python vuv.py sync --inexact
    ```
    - To use raw `uv` binary, you must setup `UV_PROJECT_ENVIRONMENT` environment variable. see [here](https://docs.astral.sh/uv/configuration/environment/#uv_project_environment)

4. **Import and use core functionality**:
    ```python
    import time

    from owa.registry import CALLABLES, LISTENERS, activate_module

    # Activate the standard environment module
    activate_module("owa.env.std")

    def callback():
        # Get current time in nanoseconds
        time_ns = CALLABLES["clock.time_ns"]()
        print(f"Current time in nanoseconds: {time_ns}")

    # Create a listener for clock/tick event
    tick = LISTENERS["clock/tick"](callback)

    # Set listener to trigger every 1 second
    tick.configure(interval=1)
    # Start the listener
    tick.start()

    # Allow the listener to run for 2 seconds
    time.sleep(2)

    # Stop the listener and wait for it to finish
    tick.stop(), tick.join()
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
