"""
Shared test fixtures for owa-core tests.
"""

from unittest.mock import Mock

import pytest
from lazyregistry import Registry


@pytest.fixture
def isolated_registries():
    """Create isolated registries for testing without affecting global state."""
    callables_registry = Registry(name="callables")
    listeners_registry = Registry(name="listeners")
    runnables_registry = Registry(name="runnables")

    return {
        "callables": callables_registry,
        "listeners": listeners_registry,
        "runnables": runnables_registry,
    }


@pytest.fixture
def create_mock_entry_point():
    """Create a mock entry point for testing."""

    def _create_mock_entry_point(name: str, load_return_value):
        mock_entry_point = Mock()
        mock_entry_point.name = name
        mock_entry_point.load.return_value = load_return_value
        return mock_entry_point

    return _create_mock_entry_point


@pytest.fixture
def mock_entry_points_factory():
    """Create a mock entry points function for testing."""

    def _mock_entry_points_factory(entry_points_list):
        def mock_entry_points(group=None):
            return entry_points_list if group == "owa.msgs" else []

        return mock_entry_points

    return _mock_entry_points_factory
