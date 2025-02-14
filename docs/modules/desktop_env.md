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
