# Desktop Environment

The Desktop Environment module (owa.env.desktop) extends Open World Agents by providing functionalities that interact with the operating system's desktop. It focuses on user interface interactions and input simulation.

## Features

- **Screen Capture:** Capture the current screen using CALLABLES["screen.capture"].
- **Window Management:** Retrieve information about active windows and search for windows by title using functions like CALLABLES["window.get_active_window"] and CALLABLES["window.get_window_by_title"].
- **Input Simulation:** Simulate mouse actions (e.g., CALLABLES["mouse.click"]) and set up keyboard listeners to handle input events.

## Usage

To activate the Desktop Environment module, include the following in your code:

```python
activate_module("owa.env.desktop")
```

After activation, you can access desktop functionalities via the global registries. For example:

```python
print(CALLABLES["screen.capture"]().shape)  # Capture and display screen dimensions
print(CALLABLES["window.get_active_window"])()  # Retrieve the active window
```

This module is essential for applications that require integration with desktop UI elements and user input simulation.

## Implementation Details

To see detailed implementation, skim over [owa_env_desktop](https://github.com/open-world-agents/open-world-agents/tree/main/projects/owa-env-desktop/owa_env_desktop). API documentation is currently being developed.

## Available Functions

### Mouse Functions
- `mouse.click` - Simulate a mouse click
- `mouse.move` - Move the mouse cursor to specified coordinates
- `mouse.position` - Get the current mouse position
- `mouse.press` - Simulate pressing a mouse button
- `mouse.release` - Simulate releasing a mouse button
- `mouse.scroll` - Simulate mouse wheel scrolling

### Keyboard Functions
- `keyboard.press` - Simulate pressing a keyboard key
- `keyboard.release` - Simulate releasing a keyboard key
- `keyboard.type` - Type a string of characters

### Screen Functions
- `screen.capture` - Capture the current screen (Note: This module utilizes `bettercam`. For better performance and extensibility, use `owa-env-gst`'s functions instead)

### Window Functions
- `window.get_active_window` - Get the currently active window
- `window.get_window_by_title` - Find a window by its title
- `window.when_active` - Run a function when a specific window becomes active

## Available Listeners

- `keyboard` - Listen for keyboard events
- `mouse` - Listen for mouse events