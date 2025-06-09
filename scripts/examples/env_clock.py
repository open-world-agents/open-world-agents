import time

from owa.core.registry import CALLABLES, LISTENERS

print(CALLABLES, LISTENERS)
# {'clock.time_ns': <built-in function time_ns>} {'clock/tick': <class 'owa.env.std.clock.ClockTickListener'>}

# Testing the clock/tick listener
tick = LISTENERS["clock/tick"]().configure(callback=lambda: print(CALLABLES["clock.time_ns"]()), interval=1)
tick.start()

time.sleep(2)  # The listener prints the current time in nanoseconds a few times

tick.stop(), tick.join()
