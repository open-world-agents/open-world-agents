import time

from owa.core.registry import CALLABLES, LISTENERS, activate_module

# Initial registry state (empty)
print(CALLABLES, LISTENERS)  # {}, {}

# Activate the standard module to register clock functionalities
activate_module("owa.env.std")
print(CALLABLES, LISTENERS)
# {'clock.time_ns': <built-in function time_ns>} {'clock/tick': <class 'owa.env.std.clock.ClockTickListener'>}

# Testing the clock/tick listener
tick = LISTENERS["clock/tick"]().configure(callback=lambda: print(CALLABLES["clock.time_ns"]()), interval=1)
tick.start()

time.sleep(2)  # The listener prints the current time in nanoseconds a few times

tick.stop(), tick.join()
