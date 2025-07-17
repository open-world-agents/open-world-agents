"""
Tests for the component access API (owa.core.component_access).
"""

from unittest.mock import patch

import pytest

from owa.core.component_access import (
    _get_all_component_names,
    get_component,
    get_component_info,
    get_namespace_components,
    get_registry,
    list_components,
)
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


class TestGetNamespaceComponents:
    """Test get_namespace_components function."""

    def test_get_namespace_components(self, isolated_registries):
        """Test getting components from a specific namespace."""
        test_registry = isolated_registries["callables"]

        # Register test components
        def test_add(a, b):
            return a + b

        def test_multiply(a, b):
            return a * b

        test_registry.register("example/add", obj_or_import_path=test_add, is_instance=True)
        test_registry.register("example/multiply", obj_or_import_path=test_multiply, is_instance=True)
        test_registry.register("other/subtract", obj_or_import_path="operator:sub")

        # Test getting namespace components
        example_components = get_namespace_components(test_registry, "example")

        assert "add" in example_components
        assert "multiply" in example_components
        assert "subtract" not in example_components  # This is in 'other' namespace

        # Test that components work
        assert example_components["add"](5, 3) == 8
        assert example_components["multiply"](4, 6) == 24

    def test_get_namespace_components_empty_namespace(self, isolated_registries):
        """Test getting components from a namespace that doesn't exist."""
        test_registry = isolated_registries["callables"]

        # Register a component in a different namespace
        test_registry.register("example/add", obj_or_import_path="operator:add")

        # Test getting from non-existent namespace
        empty_components = get_namespace_components(test_registry, "nonexistent")
        assert empty_components == {}

    def test_get_namespace_components_no_components(self, isolated_registries):
        """Test getting components when registry is empty."""
        test_registry = isolated_registries["callables"]

        # Test getting from empty registry
        empty_components = get_namespace_components(test_registry, "example")
        assert empty_components == {}


class TestGetComponentInfo:
    """Test get_component_info function."""

    def test_get_component_info_all_components(self, isolated_registries):
        """Test getting info for all components."""
        test_registry = isolated_registries["callables"]

        # Register test components (mix of loaded and unloaded)
        def test_add(a, b):
            return a + b

        test_registry.register("example/add", obj_or_import_path=test_add, is_instance=True)
        test_registry.register("example/multiply", obj_or_import_path="operator:mul")

        with patch("owa.core.component_access.CALLABLES", test_registry):
            info = get_component_info("callables")

            assert "example/add" in info
            assert "example/multiply" in info

            # Check info structure
            add_info = info["example/add"]
            assert add_info["name"] == "example/add"
            assert add_info["loaded"] is True  # Should be loaded since it's an instance
            assert add_info["import_path"] is None  # No import path for instances

            multiply_info = info["example/multiply"]
            assert multiply_info["name"] == "example/multiply"
            assert multiply_info["loaded"] is False  # Should not be loaded yet
            assert multiply_info["import_path"] == "operator:mul"

    def test_get_component_info_with_namespace_filter(self, isolated_registries):
        """Test getting info for components in a specific namespace."""
        test_registry = isolated_registries["callables"]

        # Register components in different namespaces
        test_registry.register("example/add", obj_or_import_path="operator:add")
        test_registry.register("example/multiply", obj_or_import_path="operator:mul")
        test_registry.register("other/subtract", obj_or_import_path="operator:sub")

        with patch("owa.core.component_access.CALLABLES", test_registry):
            info = get_component_info("callables", namespace="example")

            assert "example/add" in info
            assert "example/multiply" in info
            assert "other/subtract" not in info

    def test_get_component_info_invalid_component_type(self):
        """Test getting info for invalid component type."""
        info = get_component_info("invalid_type")
        assert info == {}

    def test_get_component_info_empty_registry(self, isolated_registries):
        """Test getting info from empty registry."""
        test_registry = isolated_registries["callables"]

        with patch("owa.core.component_access.CALLABLES", test_registry):
            info = get_component_info("callables")
            assert info == {}


class TestGetAllComponentNames:
    """Test _get_all_component_names function."""

    def test_get_all_component_names(self, isolated_registries):
        """Test getting all component names from registry."""
        test_registry = isolated_registries["callables"]

        # Register components (mix of loaded and unloaded)
        def test_add(a, b):
            return a + b

        test_registry.register("example/add", obj_or_import_path=test_add, is_instance=True)
        test_registry.register("example/multiply", obj_or_import_path="operator:mul")

        names = _get_all_component_names(test_registry)

        assert "example/add" in names
        assert "example/multiply" in names
        assert len(names) == 2

    def test_get_all_component_names_empty_registry(self, isolated_registries):
        """Test getting names from empty registry."""
        test_registry = isolated_registries["callables"]

        names = _get_all_component_names(test_registry)
        assert names == []

    def test_get_all_component_names_loaded_and_unloaded(self, isolated_registries):
        """Test that function returns both loaded and unloaded component names."""
        test_registry = isolated_registries["callables"]

        # Register an import path
        test_registry.register("example/multiply", obj_or_import_path="operator:mul")

        # Load the component to put it in _registry
        _ = test_registry["example/multiply"]

        # Register another component that won't be loaded
        test_registry.register("example/add", obj_or_import_path="operator:add")

        names = _get_all_component_names(test_registry)

        # Should include both loaded and unloaded
        assert "example/multiply" in names
        assert "example/add" in names
        assert len(names) == 2


class TestGetComponentEdgeCases:
    """Test edge cases for get_component function."""

    def test_get_component_all_components(self, isolated_registries):
        """Test getting all components without namespace or name."""
        test_registry = isolated_registries["callables"]

        # Register test components
        def test_add(a, b):
            return a + b

        test_registry.register("example/add", obj_or_import_path=test_add, is_instance=True)
        test_registry.register("example/multiply", obj_or_import_path="operator:mul")

        with patch("owa.core.component_access.CALLABLES", test_registry):
            all_components = get_component("callables")

            assert isinstance(all_components, dict)
            assert "example/add" in all_components
            assert "example/multiply" in all_components

            # Test that components work
            assert all_components["example/add"](5, 3) == 8

    def test_get_component_namespace_only(self, isolated_registries):
        """Test getting all components in a namespace."""
        test_registry = isolated_registries["callables"]

        # Register test components
        test_registry.register("example/add", obj_or_import_path="operator:add")
        test_registry.register("example/multiply", obj_or_import_path="operator:mul")
        test_registry.register("other/subtract", obj_or_import_path="operator:sub")

        with patch("owa.core.component_access.CALLABLES", test_registry):
            example_components = get_component("callables", namespace="example")

            assert isinstance(example_components, dict)
            assert "add" in example_components
            assert "multiply" in example_components
            assert "subtract" not in example_components

    def test_get_component_none_result(self, isolated_registries):
        """Test get_component when registry.get returns None."""
        test_registry = isolated_registries["callables"]

        with patch("owa.core.component_access.CALLABLES", test_registry):
            # Test with non-existent component
            with pytest.raises(KeyError, match="Component 'nonexistent/component' not found"):
                get_component("callables", namespace="nonexistent", name="component")


class TestListComponentsEdgeCases:
    """Test edge cases for list_components function."""

    def test_list_components_all_types(self, isolated_registries):
        """Test listing all component types."""
        # Register components in different registries
        isolated_registries["callables"].register("example/add", obj_or_import_path="operator:add")
        isolated_registries["listeners"].register("example/listener", obj_or_import_path="time:sleep")
        isolated_registries["runnables"].register("example/runnable", obj_or_import_path="time:sleep")

        with patch("owa.core.component_access.CALLABLES", isolated_registries["callables"]):
            with patch("owa.core.component_access.LISTENERS", isolated_registries["listeners"]):
                with patch("owa.core.component_access.RUNNABLES", isolated_registries["runnables"]):
                    result = list_components()

                    assert "callables" in result
                    assert "listeners" in result
                    assert "runnables" in result

                    assert "example/add" in result["callables"]
                    assert "example/listener" in result["listeners"]
                    assert "example/runnable" in result["runnables"]

    def test_list_components_with_none_registry(self):
        """Test list_components when registry is None."""
        with patch("owa.core.component_access.get_registry", return_value=None):
            result = list_components("invalid_type")
            assert result == {}

    def test_list_components_namespace_filter_no_matches(self, isolated_registries):
        """Test list_components with namespace filter that has no matches."""
        test_registry = isolated_registries["callables"]
        test_registry.register("example/add", obj_or_import_path="operator:add")

        with patch("owa.core.component_access.CALLABLES", test_registry):
            result = list_components("callables", namespace="nonexistent")

            assert "callables" in result
            assert result["callables"] == []
