import time

import pytest

from owa.core.registry import RUNNABLES, activate_module


# Automatically activate the desktop module for all tests in this session.
@pytest.fixture(scope="session", autouse=True)
def activate_owa_desktop():
    activate_module("owa.env.gst")


def test_screen_capture():
    recorder = RUNNABLES["owa.env.gst/omnimodal/subprocess_recorder"]()
    recorder.configure(filesink_location="tmp/output.mkv", window_name="open-world-agents")
    recorder.start()
    time.sleep(2)
    recorder.stop()
    recorder.join()
