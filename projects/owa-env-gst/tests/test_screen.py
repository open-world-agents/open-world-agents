import time

import pytest

from owa.registry import CALLABLES, LISTENERS, activate_module


# Automatically activate the desktop module for all tests in this session.
@pytest.fixture(scope="session", autouse=True)
def activate_owa_desktop():
    activate_module("owa_env_gst")


def test_screen_capture():
    # Test that the screen capture returns an image with the expected dimensions.
    def callback(frame, listener):
        assert frame.frame_arr.ndim == 3 and frame.frame_arr.shape[2] == 4, "Expected 4-color channel image"
        print(frame.frame_arr.shape, listener.fps, listener.latency)

    screen_listener = LISTENERS["screen"](callback)
    screen_listener.configure()
    screen_listener.start()
    time.sleep(2)
    screen_listener.stop()
    screen_listener.join()
