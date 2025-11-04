"""
Tests for the registry system (owa.core.registry).
"""

import pytest
from lazyregistry import ImportString, Registry

from owa.core.registry import CALLABLES, LISTENERS, RUNNABLES


class TestRegistry:
    """Test cases for basic Registry class."""

    def test_registry_initialization(self):
        """Test that registry initializes correctly."""
        registry = Registry(name="test")
        assert len(registry.data) == 0
        assert registry.name == "test"

    def test_register_and_access(self):
        """Test registering and accessing objects."""
        registry = Registry(name="test")

        def test_func():
            return "test"

        registry["test"] = test_func
        assert "test" in registry
        assert registry["test"] is test_func
        assert registry.get("test") is test_func

    def test_get_with_default(self):
        """Test get() method with default values."""
        registry = Registry(name="test")

        def default_func():
            return "default"

        # Non-existent with default
        result = registry.get("nonexistent", default_func)
        assert result is default_func

        # Non-existent without default
        result = registry.get("nonexistent")
        assert result is None

    def test_extend(self):
        """Test extending registry with another registry."""
        registry1 = Registry(name="test1")
        registry2 = Registry(name="test2")

        def func1():
            return "func1"

        def func2():
            return "func2"

        registry1["func1"] = func1
        registry2["func2"] = func2

        # Manually extend by copying items
        for key in registry2.keys():
            registry1[key] = registry2[key]

        assert "func1" in registry1
        assert "func2" in registry1
        assert registry1["func1"] is func1
        assert registry1["func2"] is func2


class TestLazyImportRegistry:
    """Test cases for lazy import functionality."""

    def test_lazy_import_registry_inheritance(self):
        """Test that Registry has all expected methods."""
        registry = Registry(name="callables")

        # Test that it has Registry methods
        assert hasattr(registry, "__setitem__")
        assert hasattr(registry, "__getitem__")
        assert hasattr(registry, "__contains__")
        assert hasattr(registry, "get")
        assert hasattr(registry, "data")
        assert hasattr(registry, "name")

        # Test registry name
        assert registry.name == "callables"

    def test_register_instance(self):
        """Test registering pre-loaded instances."""
        registry = Registry(name="callables")

        def test_func():
            return "test"

        registry["test"] = test_func
        assert "test" in registry
        assert registry["test"] is test_func

    def test_register_import_path(self):
        """Test registering import paths for lazy loading."""
        registry = Registry(name="callables")

        # Register with import path
        registry["operator_add"] = "operator:add"

        # Check it's stored as ImportString (not loaded yet)
        assert "operator_add" in registry
        assert isinstance(registry.data["operator_add"], ImportString)

        # Access should trigger loading
        add_func = registry["operator_add"]
        import operator

        assert add_func is operator.add
        # After loading, it's no longer an ImportString
        assert not isinstance(registry.data["operator_add"], ImportString)

    def test_eager_loading(self):
        """Test eager loading at registration time."""
        registry = Registry(name="callables")

        # Enable eager loading via attribute
        registry.eager_load = True
        registry["operator_sub"] = "operator:sub"

        # Should be loaded immediately (not an ImportString)
        assert not isinstance(registry.data["operator_sub"], ImportString)
        import operator

        assert registry["operator_sub"] is operator.sub

    def test_load_component_error_handling(self):
        """Test error handling during component loading."""
        registry = Registry(name="callables")

        # Register invalid import path
        registry["invalid"] = "nonexistent.module:function"

        # Should raise error when accessed (ValidationError from pydantic)
        with pytest.raises(Exception):  # lazyregistry raises ValidationError
            registry["invalid"]

    def test_namespace_name_pattern(self):
        """Test that the namespace/name pattern works correctly."""
        registry = Registry(name="callables")

        # Register components with namespace/name pattern
        def test_func():
            return "test"

        registry["example/test"] = test_func
        registry["other/test"] = test_func
        registry["example/other"] = test_func

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


class TestGlobalRegistries:
    """Test cases for global registry instances."""

    def test_global_registries_exist(self):
        """Test that global registries are properly initialized."""
        assert isinstance(CALLABLES, Registry)
        assert isinstance(LISTENERS, Registry)
        assert isinstance(RUNNABLES, Registry)

        assert CALLABLES.name == "callables"
        assert LISTENERS.name == "listeners"
        assert RUNNABLES.name == "runnables"
