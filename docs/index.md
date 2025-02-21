# Welcome to Open World Agents

### Everything you need to build state-of-the-art foundation multimodal desktop agent, end-to-end.

Streamline your agent's lifecycle with Open World Agents. From data capture to real-time evaluation, everything is designed for flexibility and performance.

With open-world-agents, you can:

- **Data Collection**: Collect your own desktop data, which contains timestamp-aligned **keyboard/mouse** control and **high-frequency(60Hz+) screen** data.
    - Powered by Windows APIs (`DXGI/WGC`) and the robust [GStreamer](https://gstreamer.freedesktop.org/) framework, ensuring superior performance compared to alternatives. [Learn more...](recorder/why.md)
- **Asynchronous, real-time event processing**: Compared to existing LLM-agent frameworks and [gymnasium.Env](https://gymnasium.farama.org/api/env/), our platform features an asynchronous processing design leveraging `Callables`, `Listeners`, and `Runnables`. [Learn more...](env/architecture.md)
- **Dynamic EnvPlugin Registration**: Seamlessly register and activate EnvPlugins—consisting of `Callables`, `Listeners`, and `Runnables`—at runtime to customize and extend functionality. [Learn more...](env/install_and_usage.md)
- **Extensible Design**: Easy to add custom EnvPlugin and extend functionality. [Learn more...](env/custom_plugins.md)
- **Comprehensive Examples**: We provides various examples that demonstrates how to build foundation multimodal desktop agent. Since it's just a example, you may customize anything you want.
<!-- - **Cross-Platform**: Works on Windows and macOS. -->

## Quick Start

(TODO)

## Contributing

We welcome contributions! Please see our [Contributing Guide](contributing.md) for details on how to:

- Set up your development environment.
- Submit bug reports.
- Propose new features.
- Create pull requests.

## License

This project is released under the MIT License. See the [LICENSE](https://github.com/open-world-agents/open-world-agents/blob/main/LICENSE) file for details.
