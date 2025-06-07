"""Pytest configuration for owa-env-example tests."""

import os

import pytest

# Enable auto-discovery during tests
os.environ["OWA_ENABLE_AUTO_DISCOVERY_IN_TESTS"] = "1"

from owa.core.registry import CALLABLES, LISTENERS, RUNNABLES


@pytest.fixture(scope="session")
def example_registries():
    """Provide access to the global registries for testing."""
    return {
        "callables": CALLABLES,
        "listeners": LISTENERS,
        "runnables": RUNNABLES,
    }
