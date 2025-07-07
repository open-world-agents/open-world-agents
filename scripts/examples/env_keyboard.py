from owa.core import CALLABLES, LISTENERS, MESSAGES

# Access message types through the global registry
KeyboardEvent = MESSAGES["desktop/KeyboardEvent"]

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
