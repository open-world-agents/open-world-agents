import time

import pytest

from owa.core.registry import CALLABLES, LISTENERS, activate_module


# Automatically activate the desktop module for all tests in this session.
@pytest.fixture(scope="session", autouse=True)
def activate_owa_desktop():
    activate_module("owa.env.std")


@pytest.mark.timeout(2)
def test_clock():
    assert "clock.time_ns" in CALLABLES
    assert "clock/tick" in LISTENERS

    def callback():
        called_time.append(CALLABLES["clock.time_ns"]())
        print(called_time)

    tick = LISTENERS["clock/tick"]().configure(callback=callback, interval=1)

    called_time = []
    tick.start()

    time.sleep(1.5)

    tick.stop()
    tick.join()

    assert len(called_time) == 2  # t=0 and t=1
    # check if time is not far from the expected time, within 5% error
    now = called_time[-1]
    for ct in called_time[-1::-1]:
        assert now - ct <= 1_000_000_000 * 1.05, f"{now - ct} > {1_000_000_000 * 1.05}"
        now = ct
