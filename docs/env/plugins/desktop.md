# Desktop Environment

The Desktop Environment module (owa.env.desktop) extends Open World Agents by providing functionalities that interact with the operating system's desktop. It focuses on user interface interactions and input simulation.

## Features

- **Screen Capture:** Capture the current screen using CALLABLES["desktop/screen.capture"].
- **Window Management:** Retrieve information about active windows and search for windows by title using functions like CALLABLES["desktop/window.get_active_window"] and CALLABLES["desktop/window.get_window_by_title"].
- **Input Simulation:** Simulate mouse actions (e.g., CALLABLES["desktop/mouse.click"]) and set up keyboard listeners to handle input events.

## Usage

The Desktop Environment module is automatically available when you install `owa-env-desktop`. No manual activation needed!

```python
# Components automatically available after installation
from owa.core.registry import CALLABLES, LISTENERS
```

You can access desktop functionalities via the global registries using the unified `namespace/name` pattern:

```python
print(CALLABLES["desktop/screen.capture"]().shape)  # Capture and display screen dimensions
print(CALLABLES["desktop/window.get_active_window"]())  # Retrieve the active window
```

This module is essential for applications that require integration with desktop UI elements and user input simulation.

## Implementation Details

To see detailed implementation, skim over [owa-env-desktop](https://github.com/open-world-agents/open-world-agents/tree/main/projects/owa-env-desktop). API documentation is currently being developed.

## Raw Mouse Input (High-Definition)

The desktop environment now supports raw mouse input capture for high-definition mouse data, essential for gaming applications and precision mouse tracking.

### Features

- **High-Definition Data**: Access to full mouse resolution (800+ DPI) vs standard 400 DPI
- **Unfiltered Input**: Direct access from HID stack without Windows pointer acceleration
- **Sub-pixel Precision**: Movement data not limited to screen resolution
- **Windows Support**: Uses Windows WM_INPUT messages for optimal performance

### Usage Example

```python
from owa.core import LISTENERS, MESSAGES

# Get the raw mouse event message type
RawMouseEvent = MESSAGES["desktop/RawMouseEvent"]

def on_raw_mouse_event(event):
    print(f"Raw movement: dx={event.dx}, dy={event.dy}")

    # Check for button events
    if event.button_flags & 0x0001:  # Left button down
        print("Left button pressed")
    if event.button_flags & 0x0400:  # Mouse wheel
        print(f"Wheel delta: {event.button_data}")

# Create and start the raw mouse listener
raw_mouse_listener = LISTENERS["desktop/raw_mouse"]()
raw_mouse_listener.configure(callback=on_raw_mouse_event)
raw_mouse_listener.start()

# Stop when done
raw_mouse_listener.stop()
```

### Platform Support

- **Windows**: Full support using WM_INPUT API
- **Linux/macOS**: Graceful degradation (returns False on start)

### Message Format

The `RawMouseEvent` message contains:
- `dx`, `dy`: Raw movement deltas (not limited by screen resolution)
- `button_flags`: Raw button state flags from Windows RAWMOUSE structure
- `button_data`: Additional data (wheel delta, x-button info)
- `device_handle`: Optional raw input device handle
- `timestamp`: Optional timestamp in nanoseconds

## Available Functions

### Mouse Functions
- `desktop/mouse.click` - Simulate a mouse click
- `desktop/mouse.move` - Move the mouse cursor to specified coordinates
- `desktop/mouse.position` - Get the current mouse position
- `desktop/mouse.press` - Simulate pressing a mouse button
- `desktop/mouse.release` - Simulate releasing a mouse button
- `desktop/mouse.scroll` - Simulate mouse wheel scrolling

### Mouse Listeners
- `desktop/mouse` - Standard mouse event listener (movement, clicks, scroll)
- `desktop/raw_mouse` - **NEW**: Raw mouse input listener for high-definition mouse data
- `desktop/mouse_state` - Periodic mouse state reporting

### Keyboard Functions
- `desktop/keyboard.press` - Simulate pressing a keyboard key
- `desktop/keyboard.release` - Simulate releasing a keyboard key
- `desktop/keyboard.type` - Type a string of characters
- `desktop/keyboard.press_repeat` - Simulate repeat-press when pressing key long time

### Screen Functions
- `desktop/screen.capture` - Capture the current screen (Note: This module utilizes `bettercam`. For better performance and extensibility, use `owa-env-gst`'s functions instead)

### Window Functions
- `desktop/window.get_active_window` - Get the currently active window
- `desktop/window.get_window_by_title` - Find a window by its title
- `desktop/window.when_active` - Run a function when a specific window becomes active

## Available Listeners

- `desktop/keyboard` - Listen for keyboard events
- `desktop/mouse` - Listen for mouse events


## Misc

### Library Selection Rationale
This module utilizes `pynput` for input simulation after evaluating several alternatives:

- **Why not PyAutoGUI?** Though widely used, [PyAutoGUI](https://github.com/asweigart/pyautogui) uses deprecated Windows APIs (`keybd_event/mouse_event`) rather than the modern `SendInput` method. These older APIs fail in DirectX applications and games. Additionally, PyAutoGUI has seen limited maintenance (last significant update was over 2 years ago).

- **Alternative Solutions:** Libraries like [pydirectinput](https://github.com/learncodebygaming/pydirectinput) and [pydirectinput_rgx](https://github.com/ReggX/pydirectinput_rgx) address the Windows API issue by using `SendInput`, but they lack input capturing capabilities which are essential for our use case.

- **Other Options:** We also evaluated [keyboard](https://github.com/boppreh/keyboard) and [mouse](https://github.com/boppreh/mouse) libraries but found them inadequately maintained with several unresolved bugs that could impact reliability.

### Input Auto-Repeat Functionality
For simulating key auto-repeat behavior, use the dedicated function:

```python
CALLABLES["desktop/keyboard.press_repeat"](key, press_time: float, initial_delay: float = 0.5, repeat_delay: float = 0.033)
```

This function handles the complexity of simulating hardware auto-repeat, with configurable initial delay before repeating starts and the interval between repeated keypresses.

## Auto-generated documentation

::: desktop
    handler: owa