"""Pytest configuration for owa-env-example tests."""

import pytest

from owa.core.registry import activate_module


@pytest.fixture(scope="session", autouse=True)
def activate_example_plugin():
    """Automatically activate the example plugin for all tests."""
    activate_module("owa.env.example")
