import time

from owa.registry import CALLABLES, LISTENERS, activate_module

activate_module("owa.env.std")


def test_clock():
    assert "clock.time_ns" in CALLABLES
    assert "clock/tick" in LISTENERS

    tick = LISTENERS["clock/tick"](callback)

    called_time = []

    def callback():
        called_time.append(CALLABLES["clock.time_ns"]())

    tick.configure(interval=1)
    tick.activate()

    time.sleep(3)

    tick.deactivate()
    tick.shutdown()

    assert len(called_time) >= 1
    # check if time is not far from the expected time, within 5% error
    now = called_time[-1]
    for ct in called_time[-1::-1]:
        assert now - ct <= 1_000_000_000 * 1.05, f"{now - ct} > {1_000_000_000 * 1.05}"
        now = ct
