import pytest

from owa.core.registry import CALLABLES, activate_module


# Automatically activate the desktop module for all tests in this session.
@pytest.fixture(scope="session", autouse=True)
def activate_owa_desktop():
    activate_module("owa.env.example")


def test_screen_capture():
    example_print = CALLABLES["example/print"]
    assert example_print() == "Hello, World!"
