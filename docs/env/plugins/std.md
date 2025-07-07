# Standard Environment

Core utilities and timing functions for OWA agents.

!!! info "Installation"
    ```bash
    pip install owa-core  # Included automatically
    ```

## Components

| Component | Type | Description |
|-----------|------|-------------|
| `std/time_ns` | Callable | Get current time in nanoseconds |
| `std/tick` | Listener | Periodic callback execution |

## Usage Examples

=== "Time Functions"
    ```python
    from owa.core import CALLABLES

    # Get current time
    current_time = CALLABLES["std/time_ns"]()
    print(f"Current time: {current_time}")
    ```

=== "Periodic Tasks"
    ```python
    from owa.core import LISTENERS
    import time

    def on_tick():
        print(f"Tick: {CALLABLES['std/time_ns']()}")

    # Using context manager (recommended)
    tick = LISTENERS["std/tick"]().configure(callback=on_tick, interval=1)
    with tick.session:
        time.sleep(3)  # Prints every second for 3 seconds
    ```

=== "Manual Control"
    ```python
    # Manual start/stop control
    tick = LISTENERS["std/tick"]().configure(callback=on_tick, interval=1)
    tick.start()
    time.sleep(3)
    tick.stop()
    tick.join()
    ```

## API Reference

::: std
    handler: owa