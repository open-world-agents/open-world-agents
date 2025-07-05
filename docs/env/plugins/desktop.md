# Desktop Environment

The Desktop Environment module (owa.env.desktop) extends Open World Agents by providing functionalities that interact with the operating system's desktop. It focuses on user interface interactions and input simulation.

## Installation & Usage

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

## Available Components

### Mouse Functions
- `desktop/mouse.click` - Simulate a mouse click
- `desktop/mouse.move` - Move the mouse cursor to specified coordinates
- `desktop/mouse.position` - Get the current mouse position
- `desktop/mouse.press` - Simulate pressing a mouse button
- `desktop/mouse.release` - Simulate releasing a mouse button
- `desktop/mouse.scroll` - Simulate mouse wheel scrolling

### Keyboard Functions
- `desktop/keyboard.press` - Simulate pressing a keyboard key
- `desktop/keyboard.release` - Simulate releasing a keyboard key
- `desktop/keyboard.type` - Type a string of characters
- `desktop/keyboard.press_repeat` - Simulate repeat-press when pressing key long time

For simulating key auto-repeat behavior, use the dedicated function:

```python
CALLABLES["desktop/keyboard.press_repeat"](key, press_time: float, initial_delay: float = 0.5, repeat_delay: float = 0.033)
```

This function handles the complexity of simulating hardware auto-repeat, with configurable initial delay before repeating starts and the interval between repeated keypresses.

### Screen Functions
- `desktop/screen.capture` - Capture the current screen (Note: This module utilizes `bettercam`. For better performance and extensibility, use `owa-env-gst`'s functions instead)

### Window Functions
- `desktop/window.get_active_window` - Get the currently active window
- `desktop/window.get_window_by_title` - Find a window by its title
- `desktop/window.when_active` - Run a function when a specific window becomes active

### Listeners
- `desktop/keyboard` - Listen for keyboard events
- `desktop/mouse` - Standard mouse event listener (movement, clicks, scroll)
- `desktop/raw_mouse` - Raw mouse input listener for unfiltered mouse data
- `desktop/mouse_state` - Periodic mouse state reporting

## Raw Mouse Input

Raw mouse input capture is available to separate mouse position movement from game's center-locking and from user interactions. This enables access to unfiltered mouse movement data directly from the hardware, bypassing Windows pointer acceleration and game cursor manipulation.

## Implementation Details

To see detailed implementation, skim over [owa-env-desktop](https://github.com/open-world-agents/open-world-agents/tree/main/projects/owa-env-desktop). API documentation is currently being developed.

### Library Selection Rationale
This module utilizes `pynput` for input simulation after evaluating several alternatives:

- **Why not PyAutoGUI?** Though widely used, [PyAutoGUI](https://github.com/asweigart/pyautogui) uses deprecated Windows APIs (`keybd_event/mouse_event`) rather than the modern `SendInput` method. These older APIs fail in DirectX applications and games. Additionally, PyAutoGUI has seen limited maintenance (last significant update was over 2 years ago).

- **Alternative Solutions:** Libraries like [pydirectinput](https://github.com/learncodebygaming/pydirectinput) and [pydirectinput_rgx](https://github.com/ReggX/pydirectinput_rgx) address the Windows API issue by using `SendInput`, but they lack input capturing capabilities which are essential for our use case.

- **Other Options:** We also evaluated [keyboard](https://github.com/boppreh/keyboard) and [mouse](https://github.com/boppreh/mouse) libraries but found them inadequately maintained with several unresolved bugs that could impact reliability.

## Auto-generated documentation

::: desktop
    handler: owa