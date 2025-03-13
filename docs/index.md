# Welcome to Open World Agents

### Everything you need to build state-of-the-art foundation multimodal desktop agent, end-to-end.

Streamline your agent's lifecycle with Open World Agents. From data capture to model training and real-time evaluation, everything is designed for flexibility and performance.

With open-world-agents, you can:

- **Data Collection**: Collect your own desktop data, which contains timestamp-aligned **keyboard/mouse** control and **high-frequency(60Hz+) screen** data.
    - Powered by Windows APIs (`DXGI/WGC`) and the robust [GStreamer](https://gstreamer.freedesktop.org/) framework, ensuring superior performance compared to alternatives. [Learn more...](recorder/why.md)
- **Asynchronous, real-time event processing**: Compared to existing LLM-agent frameworks and [gymnasium.Env](https://gymnasium.farama.org/api/env/), our platform features an asynchronous processing design leveraging `Callables`, `Listeners`, and `Runnables`. [Learn more...](env/index.md)
- **Dynamic EnvPlugin Registration**: Seamlessly register and activate EnvPlugins—consisting of `Callables`, `Listeners`, and `Runnables`—at runtime to customize and extend functionality. [Learn more...](env/custom_plugins.md)
- **Extensible Design**: Easy to add custom EnvPlugin and extend functionality. [Learn more...](env/custom_plugins.md)
- **Comprehensive Examples**: We provides various examples that demonstrates how to build foundation multimodal desktop agent. Since it's just a example, you may customize anything you want.
<!-- - **Cross-Platform**: Works on Windows and macOS. -->

## Quick Start

1. Simple example of using `Callables` and `Listeners`. [Learn more...](env)
    ```python
    import time

    from owa.core.registry import CALLABLES, LISTENERS, activate_module

    # Activate the standard environment module
    activate_module("owa.env.std")

    def callback():
        # Get current time in nanoseconds
        time_ns = CALLABLES["clock.time_ns"]()
        print(f"Current time in nanoseconds: {time_ns}")

    # Create a listener for clock/tick event
    tick = LISTENERS["clock/tick"]().configure(callback=callback)

    # Set listener to trigger every 1 second
    tick.configure(interval=1)
    # Start the listener
    tick.start()

    # Allow the listener to run for 2 seconds
    time.sleep(2)

    # Stop the listener and wait for it to finish
    tick.stop(), tick.join()
    ```

2. Record your own desktop usage data by just running `recorder.exe output.mkv`. [Learn more...](recorder/install_and_usage.md)


3. How to register your custom EnvPlugin. [Learn more...](env/custom_plugins.md)
    1. Write your own code
```python
from owa.core import Listener
from owa.core.registry import LISTENERS


@LISTENERS.register("my/listener")
class MyEventListener(Listener):
    def loop(self, *, stop_event, callback):
        while not stop_event.is_set():
            event = wait_and_get_event()
            callback(event)
```
    2. Use it!
```python
from owa.core.registry import LISTENERS, activate_module

activate_module("your-own-envplugin")

def callback(event):
    print(f"Captured event: {event}")

listener = LISTENERS["my/listener"]().configure(callback=callback)
listener.configure(), listener.start()

... # Run any your own logic here. listener is being executed in background as thread(ListenerThread) or process(ListenerProcess).

# Finish it by calling stop and join
listener.stop(), listener.join()
```

<!-- TODO: add agent training lifecycle example -->

## Contributing

We welcome contributions! Please see our [Contributing Guide](contributing.md) for details on how to:

- Set up your development environment.
- Submit bug reports.
- Propose new features.
- Create pull requests.

## License

This project is released under the MIT License. See the [LICENSE](https://github.com/open-world-agents/open-world-agents/blob/main/LICENSE) file for details.
