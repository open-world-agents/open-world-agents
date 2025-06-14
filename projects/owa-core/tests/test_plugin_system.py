"""
Tests for the plugin system.

This module tests the core plugin functionality including:
- LazyImportRegistry
- Component access API
- Plugin specification system

These tests focus only on owa-core functionality and use mocks
for any external dependencies.
"""

from unittest.mock import patch

import pytest

from owa.core.component_access import get_component, get_registry, list_components
from owa.core.plugin_spec import PluginSpec
from owa.core.registry import CALLABLES, LISTENERS, RUNNABLES, LazyImportRegistry, RegistryType


@pytest.fixture
def isolated_registries():
    """Create isolated registries for testing without affecting global state."""
    callables_registry = LazyImportRegistry(RegistryType.CALLABLES)
    listeners_registry = LazyImportRegistry(RegistryType.LISTENERS)
    runnables_registry = LazyImportRegistry(RegistryType.RUNNABLES)

    return {
        "callables": callables_registry,
        "listeners": listeners_registry,
        "runnables": runnables_registry,
    }


def test_plugin_spec_creation():
    """Test PluginSpec creation and validation."""
    plugin_spec = PluginSpec(
        namespace="test",
        version="1.0.0",
        description="Test plugin",
        author="Test Author",
        components={
            "callables": {
                "add": "test.module:add_function",
                "multiply": "test.module:multiply_function",
            },
            "listeners": {
                "timer": "test.module:TimerListener",
            },
        },
    )

    assert plugin_spec.namespace == "test"
    assert plugin_spec.version == "1.0.0"
    assert "callables" in plugin_spec.components
    assert "listeners" in plugin_spec.components

    # Test component name generation
    callable_names = plugin_spec.get_component_names("callables")
    assert "test/add" in callable_names
    assert "test/multiply" in callable_names

    # Test import path retrieval
    add_path = plugin_spec.get_import_path("callables", "add")
    assert add_path == "test.module:add_function"


def test_plugin_spec_validation():
    """Test PluginSpec validation for unsupported component types."""
    plugin_spec = PluginSpec(
        namespace="test",
        version="1.0.0",
        description="Test plugin",
        components={
            "callables": {"test": "test.module:test"},
            "invalid_type": {"test": "test.module:test"},
        },
    )

    try:
        plugin_spec.validate_components()
        assert False, "Should have raised ValueError for invalid component type"
    except ValueError as e:
        assert "invalid_type" in str(e)


def test_component_access_api(isolated_registries):
    """Test the enhanced component access API using isolated registries."""
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


def test_get_registry():
    """Test the get_registry function."""
    callables_registry = get_registry("callables")
    listeners_registry = get_registry("listeners")
    runnables_registry = get_registry("runnables")
    invalid_registry = get_registry("invalid")

    assert callables_registry is CALLABLES
    assert listeners_registry is LISTENERS
    assert runnables_registry is RUNNABLES
    assert invalid_registry is None


def test_lazy_import_registry_inheritance():
    """Test that LazyImportRegistry properly inherits from Registry."""
    registry = LazyImportRegistry(RegistryType.CALLABLES)

    # Test that it has Registry methods
    assert hasattr(registry, "register")
    assert hasattr(registry, "__getitem__")
    assert hasattr(registry, "__contains__")
    assert hasattr(registry, "get")

    # Test that it has LazyImportRegistry-specific attributes
    assert hasattr(registry, "_import_paths")
    assert hasattr(registry, "_load_component")

    # Test registry type
    assert registry.registry_type == RegistryType.CALLABLES


def test_namespace_name_pattern():
    """Test that the namespace/name pattern works correctly."""
    registry = LazyImportRegistry(RegistryType.CALLABLES)

    # Register components with namespace/name pattern
    def test_func():
        return "test"

    registry.register("example/test", obj_or_import_path=test_func, is_instance=True)
    registry.register("other/test", obj_or_import_path=test_func, is_instance=True)
    registry.register("example/other", obj_or_import_path=test_func, is_instance=True)

    # Test that components are properly separated by namespace
    assert "example/test" in registry
    assert "other/test" in registry
    assert "example/other" in registry

    # Test that they don't conflict
    assert registry["example/test"] is test_func
    assert registry["other/test"] is test_func
    assert registry["example/other"] is test_func

    # Test that partial names don't match
    assert "example" not in registry
    assert "test" not in registry
