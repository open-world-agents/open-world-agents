from owa.registry import CALLABLES, LISTENERS, activate_module

print(CALLABLES, LISTENERS)

activate_module("owa.env.std")
print(CALLABLES, LISTENERS)

tick = LISTENERS["clock/tick"]()
tick.configure(callback=lambda: print(CALLABLES["clock.time_ns"]()), interval=1)
tick.activate()

import time

time.sleep(3)
# tick.deactivate()
tick.shutdown()
exit()

activate_module("owa.env.desktop")

print(CALLABLES, LISTENERS)

print(CALLABLES["screen.capture"]())

# Get the callable function for mouse click
mouse_click = CALLABLES["mouse.click"]
mouse_click(1, 2)

inventory = CALLABLES["minecraft.get_inventory"](player="Steve")


# Get the listener for keyboard
def on_keyboard_event(event_type, key):
    print(f"Keyboard event: {event_type}, {key}")


keyboard_listener = LISTENERS["keyboard"](on_keyboard_event)
keyboard_listener.start()
