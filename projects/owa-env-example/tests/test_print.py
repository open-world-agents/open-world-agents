import time

import pytest

from owa.registry import CALLABLES, LISTENERS, RUNNABLES, activate_module


# Automatically activate the desktop module for all tests in this session.
@pytest.fixture(scope="session", autouse=True)
def activate_owa_desktop():
    activate_module("owa_env_example")


def test_screen_capture():
    example_print = CALLABLES["example/print"]
    assert example_print() == "Hello, World!"
