from owa.core.registry import CALLABLES, LISTENERS, activate_module
from owa.env.desktop.msg import KeyboardEvent

# Activate the desktop module to enable UI and input capabilities
activate_module("owa.env.desktop")

# Using screen capture and window management features
print(f"{CALLABLES['screen.capture']().shape=}")  # Example output: (1080, 1920, 3)
print(f"{CALLABLES['window.get_active_window']()=}")
print(f"{CALLABLES['window.get_window_by_title']('open-world-agents')=}")

# Simulating a mouse click (left button, double click)
mouse_click = CALLABLES["mouse.click"]
mouse_click("left", 2)


# Configuring a keyboard listener
def on_keyboard_event(keyboard_event: KeyboardEvent):
    print(f"Keyboard event: {keyboard_event.event_type=}, {keyboard_event.vk=}")


keyboard_listener = LISTENERS["keyboard"]().configure(callback=on_keyboard_event)
with keyboard_listener.session:
    input("Type enter to exit.\n")
