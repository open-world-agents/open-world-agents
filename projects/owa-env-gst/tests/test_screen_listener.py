import time

import numpy as np
import pytest

from owa.core.registry import LISTENERS, activate_module


# Automatically activate the desktop module for all tests in this session.
@pytest.fixture(scope="session", autouse=True)
def activate_owa_desktop():
    activate_module("owa.env.gst")


def test_screen_capture():
    latencies = []

    # Test that the screen capture returns an image with the expected dimensions.
    def callback(frame, listener):
        assert frame.frame_arr.ndim == 3 and frame.frame_arr.shape[2] == 4, "Expected 4-color channel image"
        print(frame.frame_arr.shape, listener.fps, listener.latency)

        latencies.append(listener.latency)

    screen_listener = LISTENERS["screen"]().configure(callback=callback, fps=60)
    screen_listener.start()
    time.sleep(2)
    screen_listener.stop()
    screen_listener.join()

    # Ensure that the latency is within a reasonable range. P95 < 30ms
    print(f"P95 latency: {np.percentile(latencies, 95) * 1000:.2f}ms")
    assert np.percentile(latencies, 95) < 0.03, "Latency is too high"
