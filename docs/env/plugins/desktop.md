# Desktop Environment

Mouse, keyboard, window control, and screen capture for desktop automation.

!!! info "Installation"
    ```bash
    pip install owa-env-desktop
    ```

## Components

| Category | Component | Type | Description |
|----------|-----------|------|-------------|
| **Mouse** | `desktop/mouse.click` | Callable | Simulate mouse clicks |
| | `desktop/mouse.move` | Callable | Move cursor to coordinates |
| | `desktop/mouse.position` | Callable | Get current mouse position |
| | `desktop/mouse` | Listener | Monitor mouse events |
| | `desktop/raw_mouse` | Listener | Raw mouse input (bypasses acceleration) |
| **Keyboard** | `desktop/keyboard.press` | Callable | Press/release keys |
| | `desktop/keyboard.type` | Callable | Type text strings |
| | `desktop/keyboard.press_repeat` | Callable | Simulate key auto-repeat |
| | `desktop/keyboard` | Listener | Monitor keyboard events |
| **Screen** | `desktop/screen.capture` | Callable | Capture screen (basic) |
| **Window** | `desktop/window.get_active_window` | Callable | Get active window info |
| | `desktop/window.get_window_by_title` | Callable | Find window by title |

!!! tip "Performance Note"
    For high-performance screen capture, use **[GStreamer Environment](gst.md)** instead (6x faster).

## Usage Examples

=== "Mouse Control"
    ```python
    from owa.core import CALLABLES

    # Click and move
    CALLABLES["desktop/mouse.click"]("left", 2)  # Double-click
    CALLABLES["desktop/mouse.move"](100, 200)

    # Get position
    x, y = CALLABLES["desktop/mouse.position"]()
    print(f"Mouse at: {x}, {y}")
    ```

=== "Keyboard Control"
    ```python
    from owa.core import CALLABLES

    # Type text
    CALLABLES["desktop/keyboard.type"]("Hello World!")

    # Press keys
    CALLABLES["desktop/keyboard.press"]("ctrl+c")

    # Auto-repeat (hold key)
    CALLABLES["desktop/keyboard.press_repeat"]("space", press_time=2.0)
    ```

=== "Event Monitoring"
    ```python
    from owa.core import LISTENERS
    from owa.msgs.desktop.keyboard import KeyboardEvent

    def on_key(event: KeyboardEvent):
        print(f"Key {event.event_type}: {event.vk}")

    def on_mouse(event):
        print(f"Mouse: {event.event_type} at {event.x}, {event.y}")

    # Monitor events
    with LISTENERS["desktop/keyboard"]().configure(callback=on_key).session:
        with LISTENERS["desktop/mouse"]().configure(callback=on_mouse).session:
            input("Press Enter to stop monitoring...")
    ```

=== "Window Management"
    ```python
    from owa.core import CALLABLES

    # Get window information
    active = CALLABLES["desktop/window.get_active_window"]()
    print(f"Active window: {active}")

    # Find specific window
    window = CALLABLES["desktop/window.get_window_by_title"]("Notepad")
    if window:
        print(f"Found Notepad: {window}")
    ```

## Technical Details

### Library Selection Rationale

This module utilizes `pynput` for input simulation after evaluating several alternatives:

- **Why not PyAutoGUI?** Though widely used, [PyAutoGUI](https://github.com/asweigart/pyautogui) uses deprecated Windows APIs (`keybd_event/mouse_event`) rather than the modern `SendInput` method. These older APIs fail in DirectX applications and games. Additionally, PyAutoGUI has seen limited maintenance (last significant update was over 2 years ago).

- **Alternative Solutions:** Libraries like [pydirectinput](https://github.com/learncodebygaming/pydirectinput) and [pydirectinput_rgx](https://github.com/ReggX/pydirectinput_rgx) address the Windows API issue by using `SendInput`, but they lack input capturing capabilities which are essential for our use case.

- **Other Options:** We also evaluated [keyboard](https://github.com/boppreh/keyboard) and [mouse](https://github.com/boppreh/mouse) libraries but found them inadequately maintained with several unresolved bugs that could impact reliability.

### Raw Mouse Input

Raw mouse input capture is available to separate mouse position movement from game's center-locking and from user interactions. This enables access to unfiltered mouse movement data directly from the hardware, bypassing Windows pointer acceleration and game cursor manipulation.

### Key Auto-Repeat Functionality

For simulating key auto-repeat behavior, use the dedicated function:

```python
CALLABLES["desktop/keyboard.press_repeat"](key, press_time: float, initial_delay: float = 0.5, repeat_delay: float = 0.033)
```

This function handles the complexity of simulating hardware auto-repeat, with configurable initial delay before repeating starts and the interval between repeated keypresses.

!!! info "Implementation"
    See [owa-env-desktop source](https://github.com/open-world-agents/open-world-agents/tree/main/projects/owa-env-desktop) for detailed implementation.

## API Reference

::: desktop
    handler: owa