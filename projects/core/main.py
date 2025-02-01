from owa.registry import CALLABLES, LISTENERS

print(CALLABLES, LISTENERS)

from owa.env import register_base_pkg

register_base_pkg()
print(CALLABLES, LISTENERS)


# Get the callable function for mouse click
mouse_click = CALLABLES["mouse.click"]
mouse_click(1, 2)

inventory = CALLABLES["minecraft.get_inventory"](player="Steve")


# Get the listener for keyboard
def on_keyboard_event(event_type, key):
    print(f"Keyboard event: {event_type}, {key}")


keyboard_listener = LISTENERS["keyboard"](on_keyboard_event)
keyboard_listener.start()
