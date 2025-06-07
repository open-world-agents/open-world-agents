import pytest

from owa.core.registry import CALLABLES, LISTENERS, RUNNABLES


@pytest.fixture(scope="session")
def example_registries():
    """Provide access to the global registries for testing."""
    return {
        "callables": CALLABLES,
        "listeners": LISTENERS,
        "runnables": RUNNABLES,
    }
