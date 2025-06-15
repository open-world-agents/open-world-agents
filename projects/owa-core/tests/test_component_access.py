"""
Tests for the component access API (owa.core.component_access).
"""

from unittest.mock import patch

import pytest

from owa.core.component_access import get_component, get_registry, list_components
from owa.core.registry import CALLABLES, LISTENERS, RUNNABLES


class TestComponentAccessAPI:
    """Test cases for the component access API."""

    def test_get_registry(self):
        """Test the get_registry function."""
        callables_registry = get_registry("callables")
        listeners_registry = get_registry("listeners")
        runnables_registry = get_registry("runnables")
        invalid_registry = get_registry("invalid")

        assert callables_registry is CALLABLES
        assert listeners_registry is LISTENERS
        assert runnables_registry is RUNNABLES
        assert invalid_registry is None

    def test_get_component_with_isolated_registry(self, isolated_registries):
        """Test the get_component function using isolated registries."""
        test_registry = isolated_registries["callables"]

        # Register test components
        def test_add(a, b):
            return a + b

        def test_multiply(a, b):
            return a * b

        test_registry.register("example/add", obj_or_import_path=test_add, is_instance=True)
        test_registry.register("example/multiply", obj_or_import_path=test_multiply, is_instance=True)
        test_registry.register("other/subtract", obj_or_import_path="operator:sub")

        # Mock the global registries to use our isolated ones
        with patch("owa.core.component_access.CALLABLES", test_registry):
            # Test get_component with specific component
            add_func = get_component("callables", namespace="example", name="add")
            assert add_func(5, 3) == 8

            # Test get_component with namespace (returns all in namespace)
            example_components = get_component("callables", namespace="example")
            assert "add" in example_components
            assert "multiply" in example_components
            assert example_components["add"](10, 20) == 30

    def test_list_components_with_isolated_registry(self, isolated_registries):
        """Test the list_components function using isolated registries."""
        test_registry = isolated_registries["callables"]

        # Register test components
        def test_add(a, b):
            return a + b

        def test_multiply(a, b):
            return a * b

        test_registry.register("example/add", obj_or_import_path=test_add, is_instance=True)
        test_registry.register("example/multiply", obj_or_import_path=test_multiply, is_instance=True)
        test_registry.register("other/subtract", obj_or_import_path="operator:sub")

        # Mock the global registries to use our isolated ones
        with patch("owa.core.component_access.CALLABLES", test_registry):
            # Test list_components
            all_components = list_components("callables")
            assert "callables" in all_components
            component_names = all_components["callables"]
            assert "example/add" in component_names
            assert "example/multiply" in component_names
            assert "other/subtract" in component_names

            # Test list_components with namespace filter
            example_only = list_components("callables", namespace="example")
            example_names = example_only["callables"]
            assert "example/add" in example_names
            assert "example/multiply" in example_names
            assert "other/subtract" not in example_names

    def test_get_component_error_handling(self):
        """Test error handling in get_component."""
        # Test invalid registry type
        with pytest.raises(ValueError, match="Unknown component type"):
            get_component("invalid_type", namespace="test", name="component")

        # Test missing component (should raise KeyError)
        with pytest.raises(KeyError):
            get_component("callables", namespace="nonexistent", name="component")

    def test_list_components_error_handling(self):
        """Test error handling in list_components."""
        # Test invalid registry type
        result = list_components("invalid_type")
        assert result == {}

        # Test valid registry type should work
        result = list_components("callables")
        assert "callables" in result
        assert isinstance(result["callables"], list)
