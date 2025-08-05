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
| | `desktop/mouse.press` | Callable | Press mouse button |
| | `desktop/mouse.release` | Callable | Release mouse button |
| | `desktop/mouse.scroll` | Callable | Simulate mouse wheel scrolling |
| | `desktop/mouse.get_state` | Callable | Get current mouse position and buttons |
| | `desktop/mouse.get_pointer_ballistics_config` | Callable | Get Windows pointer ballistics settings |
| | `desktop/mouse` | Listener | Monitor mouse events |
| | `desktop/mouse_state` | Listener | Monitor mouse state changes |
| | `desktop/raw_mouse` | Listener | Raw mouse input (bypasses acceleration) |
| **Keyboard** | `desktop/keyboard.press` | Callable | Press/release keys |
| | `desktop/keyboard.type` | Callable | Type text strings |
| | `desktop/keyboard.press_repeat` | Callable | Simulate key auto-repeat |
| | `desktop/keyboard.get_keyboard_repeat_timing` | Callable | Get Windows keyboard repeat timing |
| | `desktop/keyboard` | Listener | Monitor keyboard events |
| | `desktop/keyboard_state` | Listener | Monitor keyboard state changes |
| **Screen** | `desktop/screen.capture` | Callable | Capture screen (basic) |
| **Window** | `desktop/window.get_active_window` | Callable | Get active window info |
| | `desktop/window.get_window_by_title` | Callable | Find window by title |
| | `desktop/window.get_pid_by_title` | Callable | Get process ID by window title |
| | `desktop/window.when_active` | Callable | Wait until window becomes active |
| | `desktop/window.is_active` | Callable | Check if window is active |
| | `desktop/window.make_active` | Callable | Activate/focus window |
| | `desktop/window` | Listener | Monitor window events |

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

Key auto-repeat is a Windows feature where holding down a key generates multiple key events after an initial delay. When a user presses and holds a key, Windows first waits for the **repeat delay** period, then generates repeated `WM_KEYDOWN` messages at intervals determined by the **repeat rate**.

#### How Windows Auto-Repeat Works

1. **Initial Key Press**: First `WM_KEYDOWN` message is sent immediately with repeat count = 1
2. **Repeat Delay**: System waits for the configured delay (typically 250-1000ms)
3. **Repeated Events**: Additional `WM_KEYDOWN` messages are sent at the repeat rate interval (typically 30ms)
4. **Repeat Count**: Each repeated message includes an incremented repeat count in the message parameters

**System Configuration**: Windows allows users to configure auto-repeat behavior through:
- **Repeat Delay**: Time before auto-repeat begins (0-3 scale, maps to 250ms-1000ms, default: 500ms)
- **Repeat Rate**: Frequency of repeated characters (0-31 scale, maps to ~30ms-500ms intervals, default: 30ms)

These settings can be accessed programmatically via `SystemParametersInfo` with `SPI_GETKEYBOARDDELAY` and `SPI_GETKEYBOARDSPEED` parameters.

**References**:
- [Keyboard Repeat Delay and Repeat Rate](https://learn.microsoft.com/en-us/windows/win32/inputdev/about-keyboard-input#keyboard-repeat-delay-and-repeat-rate) - Microsoft documentation on keyboard repeat behavior
- [SystemParametersInfo Function](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-systemparametersinfoa) - Windows API for keyboard repeat parameters

#### Using OWA's press_repeat Function

For simulating key auto-repeat behavior, use the dedicated function:

```python
CALLABLES["desktop/keyboard.press_repeat"](key, press_time: float, initial_delay: float = 0.5, repeat_delay: float = 0.033)
```

**Parameters:**
- `key`: The key to press and repeat
- `press_time`: Total duration to hold the key (seconds)
- `initial_delay`: Time before repeating starts (default: 0.5s, matches Windows default)
- `repeat_delay`: Interval between repeated keypresses (default: 0.033s ≈ 30ms, matches Windows default)

#### Differences from True Windows Auto-Repeat

The `press_repeat` function **approximates** Windows auto-repeat behavior but isn't identical:

**OS Auto-Repeat vs OWA Implementation:**
- **OS Auto-Repeat**: `WM_KEYDOWN` messages include repeat flag (bit 30) and repeat count
- **OWA Implementation**: Multiple `WM_KEYDOWN` messages without repeat flags (each appears as individual key press)

The difference is small and commonly ignored by applications, making this approach effective for most automation scenarios.

**Why the difference exists**: Windows provides repeat detection through `WM_KEYDOWN` message parameters, but pynput does not expose these Windows-specific details. Since the primary use case is **triggering** repeat behavior rather than detecting it, this limitation doesn't affect the functionality.

**Reference**: [WM_KEYDOWN Message](https://learn.microsoft.com/en-us/windows/win32/inputdev/wm-keydown) - Official Windows documentation for key press events and message parameters

#### Technical Details: Windows Repeat Count Behavior

The `WM_KEYDOWN` repeat count (bits 0-15) behaves differently than many developers expect:

- **Not cumulative**: Each message contains the repeat count since the last processed `WM_KEYDOWN`, not a running total
- **Usually 1**: In typical applications with fast message processing, the repeat count is almost always 1
- **Higher values possible**: Only occurs when message processing is slow enough for multiple repeats to queue up

**Example**: If you hold a key and your message loop processes messages quickly, you'll receive multiple `WM_KEYDOWN` messages each with repeat count = 1. Only when processing is delayed (e.g., by adding `Sleep(1000)` in the handler) will you see higher repeat counts like 20-30.

This design allows responsive applications to process key events immediately rather than waiting for the key release.

**Reference**: [WM_KEYDOWN repeat count behavior explained](https://stackoverflow.com/questions/44897991/wm-keydown-repeat-count) - Stack Overflow discussion with practical examples


!!! info "Implementation"
    See [owa-env-desktop source](https://github.com/open-world-agents/open-world-agents/tree/main/projects/owa-env-desktop) for detailed implementation.

## API Reference

::: desktop
    handler: owa