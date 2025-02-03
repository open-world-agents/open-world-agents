import time

from owa.registry import CALLABLES, LISTENERS, activate_module

# at first, the CALLABLES and LISTENERS are empty
print(CALLABLES, LISTENERS)  # {}, {}

# ========================================
# activate the std module, which contains the clock module
activate_module("owa.env.std")
print(CALLABLES, LISTENERS)
# {'clock.time_ns': <built-in function time_ns>} {'clock/tick': <class 'owa.env.std.clock.ClockTickListener'>}

# now, let's test the clock/tick listener
tick = LISTENERS["clock/tick"]()
tick.configure(callback=lambda: print(CALLABLES["clock.time_ns"]()), interval=1)
tick.activate()

time.sleep(1)  # during this time, the tick listener will print the time in nanoseconds 1 or 2 time
# e.g. 1738595523202614300

tick.deactivate()
tick.shutdown()

# ========================================

# activate the desktop module, which contains the keyboard/mouse, screen, window modules
activate_module("owa.env.desktop")

print(CALLABLES, LISTENERS)

print(CALLABLES["screen.capture"]().shape)  # (1080, 1920, 3)
print(CALLABLES["window.get_active_window"]())
print(CALLABLES["window.get_window_by_title"]("open-world-agents"))


# Get the callable function for mouse click
mouse_click = CALLABLES["mouse.click"]
mouse_click("left", 2)


# Get the listener for keyboard
def on_keyboard_event(event_type, key):
    print(f"Keyboard event: {event_type}, {key}")


a = LISTENERS["keyboard"]
keyboard_listener = LISTENERS["keyboard"]()
keyboard_listener.configure(on_keyboard_event)
keyboard_listener.activate()

time.sleep(5)


# also, you may register and call custom callables/listeners as you want.
# e.g.
# activate_module("owa_minecraft") # you can write your own module
# inventory = CALLABLES["minecraft.get_inventory"](player="Steve")
